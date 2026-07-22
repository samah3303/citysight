"""Analytics — counting, speed estimation, density, and heatmap generation."""

from __future__ import annotations

import time
from collections import defaultdict

import numpy as np
import cv2

from backend.config import config


class AnalyticsEngine:
    """Accumulates per-frame analytics and generates heatmaps."""

    def __init__(self):
        self.px_per_m = config.pixels_per_meter
        self.heatmap_history = config.heatmap_history
        self.density_low = config.density_low
        self.density_medium = config.density_medium

        self._heatmap_accum: np.ndarray | None = None
        self._heatmap_shape: tuple[int, int] | None = None
        self._frame_count = 0

        # Rolling averages for alert baseline
        self._vehicle_history: list[float] = []
        self._pedestrian_history: list[float] = []
        self._max_history = 300

    # ── Counting ────────────────────────────────────────────────────────

    @staticmethod
    def count_objects(tracks: list[dict]) -> dict[str, int]:
        """Return per-class counts from a list of tracked objects."""
        counts: dict[str, int] = defaultdict(int)
        for t in tracks:
            name = config.class_names.get(t["class_id"], "unknown")
            counts[name] += 1
        return dict(counts)

    # ── Speed estimation (km/h) ─────────────────────────────────────────

    @staticmethod
    def estimate_speed(tracks: list[dict]) -> dict[int, float | None]:
        """Estimate speed in km/h for each confirmed track.
        Uses pixel displacement between frames scaled to real-world meters,
        assuming config.target_fps frames per second.
        """
        speeds: dict[int, float | None] = {}
        fps = config.target_fps
        if fps <= 0:
            return {t["track_id"]: None for t in tracks}

        for t in tracks:
            v_px = t.get("velocity_px", 0)
            # velocity_px = pixels moved since last frame
            # m per frame = v_px / px_per_m
            # m/s = (v_px / px_per_m) * fps
            # km/h = m/s * 3.6
            speed_ms = (v_px / AnalyticsEngine._px_per_m_static()) * fps
            speed_kmh = speed_ms * 3.6
            # Clamp to sane range
            speeds[t["track_id"]] = max(0.0, min(speed_kmh, 250.0))
        return speeds

    @staticmethod
    def _px_per_m_static() -> float:
        return config.pixels_per_meter

    # ── Density ─────────────────────────────────────────────────────────

    def density_level(self, counts: dict[str, int]) -> str:
        """Classify pedestrian density."""
        ppl = counts.get("person", 0)
        if ppl > self.density_medium:
            return "high"
        if ppl > self.density_low:
            return "medium"
        return "low"

    # ── Heatmap ─────────────────────────────────────────────────────────

    def update_heatmap(self, tracks: list[dict], shape: tuple[int, int]) -> None:
        """Accumulate object centroids into a running heatmap."""
        if self._heatmap_accum is None or self._heatmap_shape != shape:
            h, w = shape[:2]
            self._heatmap_accum = np.zeros((h, w), dtype=np.float32)
            self._heatmap_shape = (h, w)

        frame_map = np.zeros(self._heatmap_shape, dtype=np.float32)
        for t in tracks:
            centroid = t.get("centroid")
            if centroid is None:
                continue
            cx, cy = int(centroid[0]), int(centroid[1])
            if 0 <= cx < self._heatmap_shape[1] and 0 <= cy < self._heatmap_shape[0]:
                frame_map[cy, cx] += 1.0

        # Exponential decay on accumulator
        alpha = 0.95
        self._heatmap_accum = self._heatmap_accum * alpha + frame_map

    def get_heatmap_b64(self) -> str | None:
        """Return base64 PNG of the current heatmap overlay."""
        if self._heatmap_accum is None or self._heatmap_accum.max() <= 0:
            return None

        hm = self._heatmap_accum.copy()
        hm = hm / (hm.max() + 1e-6)  # 0..1
        hm = (hm * 255).astype(np.uint8)
        coloured = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        _, buf = cv2.imencode(".png", coloured)
        import base64
        return base64.b64encode(buf).decode("utf-8")

    # ── Rolling averages for alerting ───────────────────────────────────

    def record_frame_stats(self, vehicle_count: int, pedestrian_count: int) -> None:
        self._vehicle_history.append(float(vehicle_count))
        self._pedestrian_history.append(float(pedestrian_count))
        if len(self._vehicle_history) > self._max_history:
            self._vehicle_history.pop(0)
        if len(self._pedestrian_history) > self._max_history:
            self._pedestrian_history.pop(0)
        self._frame_count += 1

    @property
    def vehicle_avg(self) -> float:
        if not self._vehicle_history:
            return 0.0
        return sum(self._vehicle_history) / len(self._vehicle_history)

    @property
    def pedestrian_avg(self) -> float:
        if not self._pedestrian_history:
            return 0.0
        return sum(self._pedestrian_history) / len(self._pedestrian_history)
