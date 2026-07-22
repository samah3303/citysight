"""Video capture + frame processing → WebSocket broadcast loop.

Runs detection + tracking + analytics on every frame and pushes
FramePayload messages to all connected WebSocket clients.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from typing import Optional

import cv2
import numpy as np

from backend.config import config
from backend.detector import Detector, CentroidTracker
from backend.analytics import AnalyticsEngine
from backend.alerting import check_and_alert
from backend.database import log_event
from backend.models import TrackedObject

logger = logging.getLogger("citysight.streamer")

# Global set of connected WebSocket clients
clients: set = set()
# Shared state for the dashboard's HTTP endpoint
shared_state: dict = {
    "frame_b64": None,
    "counts": {},
    "density": "low",
    "fps": 0.0,
    "alert": None,
    "objects": [],
}
state_lock = asyncio.Lock()


class Streamer:
    """Encapsulates the frame-processing loop."""

    def __init__(self):
        self.detector: Optional[Detector] = None
        self.tracker: Optional[CentroidTracker] = None
        self.analytics: Optional[AnalyticsEngine] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._frame_idx = 0
        self._last_event_log = time.time()
        self._last_alert_check = 0

    # ── Lifecycle ───────────────────────────────────────────────────────

    async def start(self) -> None:
        """Initialise all subsystems and launch the processing loop."""
        logger.info("Loading YOLOv8 model: %s", config.model_name)
        self.detector = Detector()
        self.tracker = CentroidTracker()
        self.analytics = AnalyticsEngine()

        source = config.video_source
        if source.isdigit():
            src = int(source)
            logger.info("Opening webcam %d", src)
        else:
            src = source
            logger.info("Opening video: %s", src)
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            logger.warning("Cannot open video source %s — using synthetic frames", src)
            self.cap = None

        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Streamer started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info("Streamer stopped")

    # ── Frame loop ──────────────────────────────────────────────────────

    async def _loop(self) -> None:
        target_dt = 1.0 / config.target_fps
        while self._running:
            loop_start = time.perf_counter()

            frame = await self._read_frame()
            if frame is None:
                await asyncio.sleep(0.5)
                continue

            processed = self._process_frame(frame)
            if processed is not None:
                await self._broadcast(processed)

            elapsed = time.perf_counter() - loop_start
            if elapsed < target_dt:
                await asyncio.sleep(target_dt - elapsed)

    async def _read_frame(self) -> np.ndarray | None:
        """Read a frame from the video source or generate a synthetic one."""
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (config.frame_width, config.frame_height))
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                # Video file ended — loop it
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.resize(frame, (config.frame_width, config.frame_height))
                    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Fallback: synthetic demo frame
        return self._synthetic_frame()

    def _synthetic_frame(self) -> np.ndarray:
        """Generate a synthetic frame with moving 'objects' for demo purposes."""
        frame = np.ones((config.frame_height, config.frame_width, 3), dtype=np.uint8) * 40
        # Draw a road
        cv2.rectangle(frame, (50, 300), (config.frame_width - 50, 550), (70, 70, 70), -1)
        # Road markings
        for x in range(100, config.frame_width - 100, 120):
            cv2.rectangle(frame, (x, 420), (x + 60, 430), (200, 200, 200), -1)
        # Moving cars (simple circles that shift with frame_idx)
        t = self._frame_idx * 0.03
        h = config.frame_height
        w = config.frame_width
        # Car 1
        x1 = int(120 + (w - 240) * (0.5 + 0.5 * np.sin(t + 0.5)))
        cv2.rectangle(frame, (x1, 380), (x1 + 40, 410), (0, 100, 255), -1)
        cv2.putText(frame, "car", (x1, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        # Car 2
        x2 = int(200 + (w - 400) * (0.5 + 0.5 * np.sin(t + 2.0)))
        cv2.rectangle(frame, (x2, 400), (x2 + 50, 435), (255, 0, 0), -1)
        cv2.putText(frame, "truck", (x2, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        # Pedestrians
        px1 = int(100 + (w - 200) * (0.5 + 0.5 * np.sin(t + 1.2)))
        cv2.circle(frame, (px1, 480), 8, (0, 255, 0), -1)
        cv2.putText(frame, "person", (px1 - 15, 468), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        px2 = int(150 + (w - 300) * (0.5 + 0.5 * np.sin(t + 3.5)))
        cv2.circle(frame, (px2, 500), 8, (0, 255, 0), -1)
        cv2.putText(frame, "person", (px2 - 15, 488), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        # Bus
        bx = int(300 + (w - 600) * (0.5 + 0.5 * np.sin(t + 4.0)))
        cv2.rectangle(frame, (bx, 350), (bx + 80, 400), (0, 180, 200), -1)
        cv2.putText(frame, "bus", (bx, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        # Motorcycle
        mx = int(80 + (w - 160) * (0.5 + 0.5 * np.sin(t + 1.7)))
        cv2.rectangle(frame, (mx, 390), (mx + 15, 400), (255, 200, 0), -1)
        cv2.putText(frame, "moto", (mx, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 0), 1)
        # Bicycle
        bix = int(200 + (w - 400) * (0.5 + 0.5 * np.cos(t + 2.5)))
        cv2.circle(frame, (bix, 460), 6, (255, 255, 0), -1)
        cv2.putText(frame, "bike", (bix - 10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)

        return frame

    # ── Frame processing ─────────────────────────────────────────────────

    def _process_frame(self, frame: np.ndarray) -> dict | None:
        """Run detection + tracking + analytics on one frame."""
        self._frame_idx += 1
        h, w = frame.shape[:2]

        try:
            detections = self.detector.detect(frame)
        except Exception as exc:
            logger.error("Detection error: %s", exc)
            detections = []

        tracks = self.tracker.update(detections, self._frame_idx)

        counts = AnalyticsEngine.count_objects(tracks)
        speeds = AnalyticsEngine.estimate_speed(tracks)
        density = self.analytics.density_level(counts)
        self.analytics.update_heatmap(tracks, (h, w))
        heatmap_b64 = self.analytics.get_heatmap_b64()

        # Build object list for payload
        objects_payload = []
        for t in tracks:
            bbox = t["bbox"]
            # Normalise bbox to 0..1
            nb = [
                bbox[0] / w, bbox[1] / h,
                bbox[2] / w, bbox[3] / h,
            ]
            objects_payload.append(TrackedObject(
                track_id=t["track_id"],
                class_id=t["class_id"],
                class_name=config.class_names.get(t["class_id"], "unknown"),
                bbox=nb,
                confidence=0.85,  # tracked, not raw detection confidence
                speed_kmh=speeds.get(t["track_id"]),
                dwell_seconds=t.get("dwell_frames", 0) / max(config.target_fps, 1),
            ))

        # Annotate frame
        annotated = self._annotate_frame(frame.copy(), tracks, speeds, counts, density)

        # Encode
        _, jpg = cv2.imencode(".jpg", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 75])
        frame_b64 = base64.b64encode(jpg).decode("utf-8")

        # FPS estimate
        now = time.perf_counter()
        if not hasattr(self, '_last_ts'):
            self._last_ts = now
        fps = 1.0 / max(now - self._last_ts, 0.001)
        self._last_ts = now

        result = {
            "frame_b64": frame_b64,
            "heatmap_b64": heatmap_b64,
            "objects": objects_payload,
            "counts": counts,
            "density": density,
            "fps": fps,
            "alert": None,
        }

        # Periodically check for alerts
        vc = counts.get("car", 0) + counts.get("truck", 0) + counts.get("bus", 0) + counts.get("motorcycle", 0)
        pc = counts.get("person", 0)
        self.analytics.record_frame_stats(vc, pc)

        alert = None
        if self._frame_idx - self._last_alert_check >= config.alert_check_frames:
            self._last_alert_check = self._frame_idx
            alert = asyncio.ensure_future(
                check_and_alert(
                    vehicle_count=vc,
                    vehicle_avg=self.analytics.vehicle_avg,
                    pedestrian_count=pc,
                    pedestrian_avg=self.analytics.pedestrian_avg,
                    density=density,
                    vehicle_types={k: v for k, v in counts.items() if k != "person"},
                )
            )
            # Fire-and-forget; result won't block frame processing
            if alert is not None:
                result["_alert_future"] = alert

        # Periodically log to DB
        now_ts = time.time()
        if now_ts - self._last_event_log >= config.event_log_interval_seconds:
            self._last_event_log = now_ts
            asyncio.ensure_future(
                log_event(
                    event_type="periodic_snapshot",
                    vehicle_count=vc,
                    pedestrian_count=pc,
                    density_level=density,
                )
            )

        # If density is high, log a high_density event
        if density == "high" and self._frame_idx % 30 == 0:
            asyncio.ensure_future(
                log_event(
                    event_type="high_density",
                    vehicle_count=vc,
                    pedestrian_count=pc,
                    density_level=density,
                    alert_message=f"High crowd density: {pc} pedestrians detected",
                )
            )

        return result

    def _annotate_frame(
        self,
        frame: np.ndarray,
        tracks: list[dict],
        speeds: dict[int, float | None],
        counts: dict[str, int],
        density: str,
    ) -> np.ndarray:
        """Draw bounding boxes, IDs, and speed labels on the frame."""
        h, w = frame.shape[:2]

        for t in tracks:
            bbox = t["bbox"]
            cls_name = config.class_names.get(t["class_id"], "?")
            tid = t["track_id"]
            speed = speeds.get(tid)

            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Colour per class
            color = self._class_color(t["class_id"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{tid}:{cls_name}"
            if speed is not None:
                label += f" {speed:.0f}km/h"
            cv2.putText(frame, label, (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Overlay counters
        y0 = 30
        for key, val in counts.items():
            cv2.putText(frame, f"{key}: {val}", (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y0 += 25

        # Density indicator
        d_color = (0, 255, 0) if density == "low" else (0, 255, 255) if density == "medium" else (0, 0, 255)
        cv2.putText(frame, f"Density: {density.upper()}", (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, d_color, 2)

        return frame

    @staticmethod
    def _class_color(cls_id: int) -> tuple[int, int, int]:
        colors = {
            0: (0, 255, 0),      # person — green
            1: (255, 255, 0),    # bicycle — cyan
            2: (0, 100, 255),    # car — orange
            3: (255, 200, 0),    # motorcycle — light blue
            5: (0, 180, 200),    # bus — gold
            7: (255, 0, 0),      # truck — red
        }
        return colors.get(cls_id, (200, 200, 200))

    # ── Broadcast ────────────────────────────────────────────────────────

    async def _broadcast(self, payload: dict) -> None:
        """Push frame data to all connected WebSocket clients."""
        # Resolve any pending alert future
        alert_text = None
        alert_future = payload.pop("_alert_future", None)
        if alert_future is not None:
            try:
                alert_text = await alert_future
            except Exception:
                alert_text = None
        payload["alert"] = alert_text

        # Update shared state for HTTP endpoints
        async with state_lock:
            shared_state["frame_b64"] = payload["frame_b64"]
            shared_state["counts"] = payload["counts"]
            shared_state["density"] = payload["density"]
            shared_state["fps"] = payload["fps"]
            shared_state["alert"] = alert_text
            shared_state["objects"] = [
                o.dict() for o in payload["objects"]
            ]

        # Serialise
        import json
        ws_data = {
            "frame_b64": payload["frame_b64"],
            "heatmap_b64": payload["heatmap_b64"],
            "objects": [o.dict() for o in payload["objects"]],
            "counts": payload["counts"],
            "density": payload["density"],
            "fps": payload["fps"],
            "alert": alert_text,
        }
        msg = json.dumps(ws_data)

        dead = set()
        for ws in clients:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.add(ws)
        clients.difference_update(dead)
