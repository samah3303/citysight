"""YOLOv8 object detection + centroid-based multi-object tracker.

Uses ultralytics YOLO for detection and a simple Kalman-IoU tracker
(similar in spirit to SORT / DeepSORT core) for tracking across frames.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

import numpy as np
from ultralytics import YOLO

from backend.config import config


# ── Detection ──────────────────────────────────────────────────────────────

class Detector:
    """YOLOv8 wrapper that only reports the target classes."""

    def __init__(self):
        self.model = YOLO(config.model_name)
        self.target = set(config.target_classes)
        self.conf = config.confidence_threshold

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Run detection on an RGB frame (H×W×3 numpy array).
        Returns list of dicts: {bbox_xyxy, confidence, class_id}.
        """
        results = self.model(frame, verbose=False)[0]
        detections: list[dict] = []
        if results.boxes is None:
            return detections
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in self.target:
                continue
            conf = float(box.conf[0])
            if conf < self.conf:
                continue
            xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2] in pixel coords
            detections.append({
                "bbox": xyxy,
                "confidence": conf,
                "class_id": cls_id,
            })
        return detections


# ── Centroid / IoU tracker ─────────────────────────────────────────────────

def _iou(bb1: tuple, bb2: tuple) -> float:
    """Intersection-over-Union between two [x1,y1,x2,y2] boxes."""
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    inter = (x_right - x_left) * (y_bottom - y_top)
    area1 = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    area2 = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    return inter / (area1 + area2 - inter + 1e-6)


class CentroidTracker:
    """Simple centroid-based multi-object tracker with IoU matching.

    When a detection matches an existing track (IoU > threshold), the
    centroid positions are used to estimate velocity for speed computation.
    """

    def __init__(self):
        self.max_age = config.tracker_max_age
        self.min_hits = config.tracker_min_hits
        self.iou_threshold = config.tracker_iou_threshold

        self.next_id = 0
        self.tracks: dict[int, dict] = {}  # track_id → state
        self._centroids: dict[int, tuple[float, float]] = {}
        self._prev_centroids: dict[int, tuple[float, float]] = {}
        self._ages: dict[int, int] = {}  # frames since last seen
        self._hits: dict[int, int] = {}  # total frames matched
        self._birth_frames: dict[int, int] = {}  # frame-of-birth for dwell

    def update(self, detections: list[dict], frame_idx: int) -> list[dict]:
        """Match detections to existing tracks and return currently active tracks.

        Each returned track dict:
            track_id, class_id, bbox, centroid, velocity_px, dwell_frames, confirmed
        """
        # If no tracks yet, initialise everything as new
        if not self.tracks:
            for det in detections:
                tid = self.next_id
                self.next_id += 1
                bbox = det["bbox"]
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                self.tracks[tid] = {
                    "bbox": bbox,
                    "class_id": det["class_id"],
                    "centroid": (cx, cy),
                }
                self._centroids[tid] = (cx, cy)
                self._ages[tid] = 0
                self._hits[tid] = 1
                self._birth_frames[tid] = frame_idx
            return self._active_tracks(frame_idx)

        # Build cost matrix (1 - IoU).  Rows = tracks, Cols = detections.
        track_ids = list(self.tracks.keys())
        n_tracks = len(track_ids)
        n_dets = len(detections)
        cost = np.full((n_tracks, n_dets), 1.0)
        for i, tid in enumerate(track_ids):
            tb = tuple(self.tracks[tid]["bbox"])
            for j, det in enumerate(detections):
                cost[i, j] = 1.0 - _iou(tb, tuple(det["bbox"]))

        # Greedy matching
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        if n_tracks > 0 and n_dets > 0:
            while True:
                i, j = np.unravel_index(cost.argmin(), cost.shape)
                if cost[i, j] > (1.0 - self.iou_threshold):
                    break
                tid = track_ids[i]
                matched_tracks.add(i)
                matched_dets.add(j)
                det = detections[j]
                bbox = det["bbox"]
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                self._prev_centroids[tid] = self._centroids.get(tid, (cx, cy))
                self._centroids[tid] = (cx, cy)
                self.tracks[tid] = {
                    "bbox": bbox,
                    "class_id": det["class_id"],
                    "centroid": (cx, cy),
                }
                self._ages[tid] = 0
                self._hits[tid] += 1
                # mask row/col
                cost[i, :] = 1.0
                cost[:, j] = 1.0

        # Age unmatched tracks
        for i, tid in enumerate(track_ids):
            if i not in matched_tracks:
                self._ages[tid] += 1

        # Create new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_dets:
                tid = self.next_id
                self.next_id += 1
                bbox = det["bbox"]
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                self.tracks[tid] = {
                    "bbox": bbox,
                    "class_id": det["class_id"],
                    "centroid": (cx, cy),
                }
                self._centroids[tid] = (cx, cy)
                self._ages[tid] = 0
                self._hits[tid] = 1
                self._birth_frames[tid] = frame_idx

        # Prune old tracks
        for tid in list(self.tracks.keys()):
            if self._ages[tid] > self.max_age:
                self.tracks.pop(tid, None)
                self._centroids.pop(tid, None)
                self._prev_centroids.pop(tid, None)
                self._ages.pop(tid, None)
                self._hits.pop(tid, None)
                self._birth_frames.pop(tid, None)

        return self._active_tracks(frame_idx)

    def _active_tracks(self, frame_idx: int) -> list[dict]:
        """Return list of confirmed (min_hits) and freshly matched tracks."""
        active = []
        for tid, track in self.tracks.items():
            if self._hits[tid] < self.min_hits and self._ages[tid] > 0:
                continue
            prev = self._prev_centroids.get(tid)
            curr = self._centroids.get(tid, track["centroid"])
            vx = vy = 0.0
            if prev is not None and curr is not None:
                vx = curr[0] - prev[0]
                vy = curr[1] - prev[1]
            dwell = frame_idx - self._birth_frames.get(tid, frame_idx)
            active.append({
                "track_id": tid,
                "class_id": track["class_id"],
                "bbox": track["bbox"],
                "centroid": curr,
                "velocity_px": math.hypot(vx, vy),
                "dwell_frames": dwell,
                "confirmed": self._hits[tid] >= self.min_hits,
            })
        return active

    def reset(self):
        """Clear all tracks (useful on video source change)."""
        self.next_id = 0
        self.tracks.clear()
        self._centroids.clear()
        self._prev_centroids.clear()
        self._ages.clear()
        self._hits.clear()
        self._birth_frames.clear()
