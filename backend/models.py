"""Pydantic models for CitySight — WebSocket messages and DB records."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


# ── Detection / tracking objects ──────────────────────────────────────────

class TrackedObject(BaseModel):
    """A single tracked object in a frame."""
    track_id: int
    class_id: int
    class_name: str
    bbox: list[float]  # [x1, y1, x2, y2] normalised 0..1
    confidence: float
    speed_kmh: float | None = None
    dwell_seconds: float | None = None


class FramePayload(BaseModel):
    """Payload pushed over WebSocket every ~100ms."""
    frame_b64: str
    heatmap_b64: str | None = None
    objects: list[TrackedObject] = []
    counts: dict[str, int] = {}  # e.g. {"car": 12, "person": 5}
    density: str = "low"  # low | medium | high
    fps: float = 0.0
    alert: str | None = None


# ── Database models ───────────────────────────────────────────────────────

class EventRecord(BaseModel):
    """Row in the events table."""
    timestamp: datetime
    event_type: str  # "periodic_snapshot", "high_density", "anomaly"
    vehicle_count: int
    pedestrian_count: int
    density_level: str
    alert_message: str | None = None
    details: dict | None = None


# ── Alert request / response ──────────────────────────────────────────────

class AlertRequest(BaseModel):
    vehicle_count: int
    vehicle_avg: float
    pedestrian_count: int
    pedestrian_avg: float
    density: str
    vehicle_types: dict[str, int] = {}

class AlertResponse(BaseModel):
    alert_text: str
    severity: str  # info | warning | critical
