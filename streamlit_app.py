"""CitySight — Streamlit Cloud deployment.
Runs YOLOv8 detection + tracking + analytics in a background thread.
Displays live video feed, charts, heatmap, and alerts directly in Streamlit.
No separate backend server needed.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2

# Inject secrets/env before importing backend modules
def _secret(key: str, default: str = "") -> str:
    try:
        import streamlit as st
        return st.secrets.get(key, os.environ.get(key, default))
    except Exception:
        return os.environ.get(key, default)

os.environ.setdefault("YOLO_MODEL", _secret("YOLO_MODEL", "yolov8n.pt"))
os.environ.setdefault("CONFIDENCE_THRESHOLD", _secret("CONFIDENCE_THRESHOLD", "0.4"))
os.environ.setdefault("VIDEO_SOURCE", _secret("VIDEO_SOURCE", "0"))
os.environ.setdefault("DEEPSEEK_API_KEY", _secret("DEEPSEEK_API_KEY", ""))
os.environ.setdefault("DATABASE_URL", _secret("DATABASE_URL", ""))

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

# Re-import config after env vars are set
from backend.config import config
from backend.detector import Detector, CentroidTracker
from backend.analytics import AnalyticsEngine
from backend.alerting import check_and_alert
from backend.database import init_db, log_event, recent_events

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("citysight")

# ── Shared state (thread-safe) ──────────────────────────────────────────
_state_lock = threading.Lock()
shared_state: dict = {
    "frame_b64": None,
    "heatmap_b64": None,
    "counts": {},
    "density": "low",
    "fps": 0.0,
    "alert": None,
    "objects": [],
    "running": False,
}

# ── Background processing thread ─────────────────────────────────────────
class CitySightProcessor(threading.Thread):
    """Runs YOLOv8 detection + tracking + analytics in a background thread."""

    def __init__(self):
        super().__init__(daemon=True)
        self.detector = None
        self.tracker = None
        self.analytics = None
        self.cap = None
        self._stop_event = threading.Event()
        self._frame_idx = 0

    def _class_color(self, cls_id: int) -> tuple:
        colors = {
            0: (0, 255, 0), 1: (255, 255, 0), 2: (0, 100, 255),
            3: (255, 200, 0), 5: (0, 180, 200), 7: (255, 0, 0),
        }
        return colors.get(cls_id, (200, 200, 200))

    def run(self):
        try:
            logger.info("Loading YOLOv8 model: %s", config.model_name)
            self.detector = Detector()
            self.tracker = CentroidTracker()
            self.analytics = AnalyticsEngine()

            source = config.video_source
            if source.replace(".", "").isdigit():
                src = int(source)
            else:
                src = source

            self.cap = cv2.VideoCapture(src)
            if not self.cap.isOpened():
                logger.warning("Cannot open video source — running in demo mode")
                self.cap = None

            target_dt = 1.0 / config.target_fps
            last_event_log = time.time()
            last_alert_check = 0

            with _state_lock:
                shared_state["running"] = True

            while not self._stop_event.is_set():
                loop_start = time.perf_counter()

                # Read frame
                frame = self._read_frame()
                if frame is None:
                    time.sleep(0.5)
                    continue

                h, w = frame.shape[:2]

                # Detection
                detections = self.detector.detect(frame)

                # Tracking
                tracks = self.tracker.update(detections, self._frame_idx)
                self._frame_idx += 1

                # Analytics
                counts = self.analytics.count_objects(tracks)
                density = self.analytics.density_level(counts)
                self.analytics.record_frame_stats(
                    sum(v for k, v in counts.items() if k != "person"),
                    counts.get("person", 0),
                )
                self.analytics.update_heatmap(tracks, (h, w))
                heatmap_b64 = self.analytics.get_heatmap_b64()

                # Draw bounding boxes
                annotated = frame.copy()
                for t in tracks:
                    if not t.get("confirmed", True):
                        continue
                    bbox = t["bbox"]
                    cls_id = t["class_id"]
                    tid = t["track_id"]
                    color = self._class_color(cls_id)
                    cv2.rectangle(annotated, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    label = f"ID:{tid} {config.class_names.get(cls_id, '?')}"
                    cv2.putText(annotated, label, (int(bbox[0]), int(bbox[1]) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Encode frame
                _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
                frame_b64 = base64.b64encode(buf).decode("utf-8")

                # Compute FPS
                elapsed = time.perf_counter() - loop_start
                fps = 1.0 / elapsed if elapsed > 0 else 0

                # Alerting (every alert_check_frames)
                alert_text = None
                now = time.time()
                if self._frame_idx - last_alert_check >= config.alert_check_frames:
                    last_alert_check = self._frame_idx
                    if config.deepseek_api_key:
                        import asyncio as _asyncio
                        try:
                            alert_text = _asyncio.new_event_loop().run_until_complete(
                                check_and_alert(
                                    sum(v for k, v in counts.items() if k != "person"),
                                    self.analytics.vehicle_avg,
                                    counts.get("person", 0),
                                    self.analytics.pedestrian_avg,
                                    density,
                                    {k: v for k, v in counts.items() if k != "person"},
                                )
                            )
                        except Exception:
                            alert_text = None

                # Update shared state
                with _state_lock:
                    shared_state.update({
                        "frame_b64": frame_b64,
                        "heatmap_b64": heatmap_b64,
                        "counts": counts,
                        "density": density,
                        "fps": fps,
                        "alert": alert_text,
                        "objects": tracks,
                    })

                # Event logging to DB (every interval)
                if config.database_url and (now - last_event_log >= config.event_log_interval_seconds):
                    last_event_log = now
                    import asyncio as _asyncio
                    try:
                        vc = sum(v for k, v in counts.items() if k != "person")
                        pc = counts.get("person", 0)
                        _asyncio.new_event_loop().run_until_complete(
                            log_event("scheduled", vc, pc, density, alert_text)
                        )
                    except Exception:
                        pass

                # Sleep to maintain target FPS
                sleep_time = target_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except Exception as e:
            logger.error("Processing thread error: %s", e)
            import traceback
            traceback.print_exc()
        finally:
            if self.cap:
                self.cap.release()
            with _state_lock:
                shared_state["running"] = False

    def _read_frame(self):
        """Read a frame from the video source, or generate a synthetic one."""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (config.frame_width, config.frame_height))
                return frame
        # Synthetic "waiting" frame
        frame = np.zeros((config.frame_height, config.frame_width, 3), dtype=np.uint8)
        cv2.putText(frame, "Waiting for video source...", (80, config.frame_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(frame, f"Source: {config.video_source}", (80, config.frame_height // 2 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 1)
        return frame

    def stop(self):
        self._stop_event.set()


# ── Page Config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CitySight — Smart City Analytics",
    page_icon="🏙️",
    layout="wide",
)

# ── Session state ────────────────────────────────────────────────────────
if "processor" not in st.session_state:
    st.session_state.processor = None
if "count_history" not in st.session_state:
    st.session_state.count_history = []
if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "processor_running" not in st.session_state:
    st.session_state.processor_running = False

# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏙️ CitySight")
    st.markdown("**Smart City Vision Analytics**")
    st.divider()

    # Start / Stop controls
    if st.session_state.processor is None:
        if st.button("▶️ Start Stream", type="primary", use_container_width=True):
            proc = CitySightProcessor()
            proc.start()
            st.session_state.processor = proc
            st.session_state.processor_running = True
            st.rerun()
    else:
        running = shared_state.get("running", False)
        st.metric("Status", "🟢 Streaming" if running else "🟡 Starting...")
        if st.button("⏹ Stop Stream", use_container_width=True):
            st.session_state.processor.stop()
            st.session_state.processor = None
            st.session_state.processor_running = False
            st.rerun()

    st.divider()
    st.markdown("### 📡 Live Stats")
    with _state_lock:
        data = dict(shared_state)
    if data.get("running"):
        counts = data.get("counts", {})
        st.metric("Vehicles", sum(v for k, v in counts.items() if k != "person"))
        st.metric("Pedestrians", counts.get("person", 0))
        st.metric("FPS", f"{data.get('fps', 0):.1f}")
        density = data.get("density", "low")
        density_icon = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(density, "⚪")
        st.metric("Density", f"{density_icon} {density.upper()}")
    else:
        st.caption("Click 'Start Stream' to begin")

    st.divider()
    st.markdown("### 🔔 Recent Alerts")
    for alert in st.session_state.alerts[-5:]:
        st.warning(alert)

    st.divider()
    st.caption("CitySight v1.0.0 | Streamlit Cloud")

# ── Main Layout ───────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("## 📹 Live Feed")
    video_placeholder = st.empty()

with col2:
    st.markdown("## 📊 Vehicle Breakdown")
    chart_placeholder = st.empty()

col3, col4 = st.columns(2)

with col3:
    st.markdown("## 📈 Counts Over Time")
    timeline_placeholder = st.empty()

with col4:
    st.markdown("## 🔥 Activity Heatmap")
    heatmap_placeholder = st.empty()

st.divider()
st.markdown("## 📋 Recent Events")
events_placeholder = st.empty()

# ── Auto-refresh ─────────────────────────────────────────────────────────
if st.session_state.processor is not None and shared_state.get("running", False):
    # Read current state
    with _state_lock:
        data = dict(shared_state)

    # Track count history
    if data.get("counts"):
        counts = data["counts"]
        ts = time.time()
        vehicles = sum(v for k, v in counts.items() if k != "person")
        pedestrians = counts.get("person", 0)
        st.session_state.count_history.append({
            "timestamp": ts, "vehicles": vehicles, "pedestrians": pedestrians,
        })
        if len(st.session_state.count_history) > 300:
            st.session_state.count_history = st.session_state.count_history[-300:]

    # Track alerts
    alert = data.get("alert")
    if alert:
        ts = datetime.now().strftime("%H:%M:%S")
        st.session_state.alerts.append(f"[{ts}] {alert}")
        if len(st.session_state.alerts) > 50:
            st.session_state.alerts = st.session_state.alerts[-50:]

    # ── Video Frame ──────────────────────────────────────────────────
    if data.get("frame_b64"):
        frame_html = f"""
        <div style="position:relative; border-radius:8px; overflow:hidden; background:#111;">
            <img src="data:image/jpeg;base64,{data['frame_b64']}"
                 style="width:100%; display:block;" />
            <div style="position:absolute; top:10px; right:10px;
                        background:rgba(0,0,0,0.7); color:#0f0;
                        padding:4px 12px; border-radius:4px; font-family:monospace;
                        font-size:13px;">
                {data.get('fps', 0):.1f} FPS | {data.get('density','low').upper()}
            </div>
        </div>
        """
        video_placeholder.markdown(frame_html, unsafe_allow_html=True)
    else:
        video_placeholder.info("Waiting for video stream...")

    # ── Vehicle Breakdown ────────────────────────────────────────────
    if data.get("counts"):
        counts = data["counts"]
        vehicle_data = {k: v for k, v in counts.items() if k != "person"}
        if vehicle_data:
            fig = go.Figure(data=[
                go.Bar(x=list(vehicle_data.keys()), y=list(vehicle_data.values()),
                       marker_color=["#FF6B35", "#FFD23F", "#3BCEAC", "#5D9CEC", "#9B5DE5"],
                       text=list(vehicle_data.values()), textposition="auto")
            ])
            fig.update_layout(
                margin=dict(l=20, r=20, t=10, b=20), height=280,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ccc"), xaxis=dict(title=None), yaxis=dict(title=None),
            )
            chart_placeholder.plotly_chart(fig, use_container_width=True)
        else:
            chart_placeholder.info("No vehicles detected")
    else:
        chart_placeholder.info("Waiting for data...")

    # ── Timeline ─────────────────────────────────────────────────────
    hist = st.session_state.count_history
    if hist:
        df = pd.DataFrame(hist)
        df["time"] = pd.to_datetime(df["timestamp"], unit="s")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df["time"], y=df["vehicles"], mode="lines", name="Vehicles",
                                   line=dict(color="#FF6B35", width=2)))
        fig2.add_trace(go.Scatter(x=df["time"], y=df["pedestrians"], mode="lines", name="Pedestrians",
                                   line=dict(color="#3BCEAC", width=2)))
        fig2.update_layout(
            margin=dict(l=20, r=20, t=10, b=20), height=280,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ccc"), legend=dict(orientation="h", yanchor="top", y=1.1),
            xaxis=dict(title=None), yaxis=dict(title=None),
        )
        timeline_placeholder.plotly_chart(fig2, use_container_width=True)
    else:
        timeline_placeholder.info("Accumulating data...")

    # ── Heatmap ──────────────────────────────────────────────────────
    if data.get("heatmap_b64"):
        heatmap_html = f"""
        <div style="border-radius:8px; overflow:hidden;">
            <img src="data:image/png;base64,{data['heatmap_b64']}" style="width:100%;" />
        </div>
        """
        heatmap_placeholder.markdown(heatmap_html, unsafe_allow_html=True)
    else:
        heatmap_placeholder.info("Heatmap accumulating...")

    # ── Events ───────────────────────────────────────────────────────
    if config.database_url:
        import asyncio as _asyncio
        try:
            loop = _asyncio.new_event_loop()
            events_data = loop.run_until_complete(recent_events(20))
            loop.close()
        except Exception:
            events_data = []
        if events_data:
            df_events = pd.DataFrame(events_data)
            cols = ["timestamp", "event_type", "vehicle_count", "pedestrian_count", "density_level", "alert_message"]
            display_cols = [c for c in cols if c in df_events.columns]
            if display_cols:
                df_display = df_events[display_cols].copy()
                if len(df_display.columns) >= 5:
                    df_display.columns = ["Time", "Type", "Vehicles", "Pedestrians", "Density"][:len(df_display.columns)]
                events_placeholder.dataframe(df_display, use_container_width=True, hide_index=True)
            else:
                events_placeholder.caption("No events logged yet.")
        else:
            events_placeholder.caption("No events logged yet (configure DATABASE_URL for persistence).")
    else:
        events_placeholder.caption("Database not configured. Set DATABASE_URL in secrets for event persistence.")

    # Auto-rerun every 2 seconds for live updates
    time.sleep(1)
    st.rerun()

else:
    st.info("👆 Click **Start Stream** in the sidebar to begin processing. "
            "The YOLOv8 model will auto-download on first run (may take a moment).")
    st.markdown("""
    ### How It Works
    1. The app spawns a background thread that captures video, runs YOLOv8 detection, and tracks objects.
    2. Results are displayed in real-time with bounding boxes, charts, and a heatmap.
    3. DeepSeek AI generates intelligent alerts when traffic deviates from baseline patterns.

    ### Configuration
    Set these in Streamlit Cloud secrets or `.env` file:
    - `YOLO_MODEL` — model file (default: yolov8n.pt, auto-downloads)
    - `VIDEO_SOURCE` — 0 for webcam, or path/URL to video file
    - `DEEPSEEK_API_KEY` — for AI-powered alerts (optional)
    - `DATABASE_URL` — Neon PostgreSQL for event persistence (optional)
    """)
