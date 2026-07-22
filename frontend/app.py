"""CitySight — Streamlit Real-time Dashboard.

Connects to the FastAPI backend via WebSocket and displays:
  - Live video feed with bounding boxes
  - Vehicle type breakdown (Plotly bar chart)
  - Counts over time (Plotly line chart)
  - Activity heatmap
  - Density gauge
  - Recent alerts
  - DB events table
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from io import BytesIO

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import httpx
import websockets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("citysight.frontend")

# ── Config ─────────────────────────────────────────────────────────────

BACKEND_HOST = "localhost"
BACKEND_PORT = 8000
BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"
WS_URL = f"ws://{BACKEND_HOST}:{BACKEND_PORT}/ws"

st.set_page_config(
    page_title="CitySight — Smart City Analytics",
    page_icon="🏙️",
    layout="wide",
)

# ── Session state initialisation ────────────────────────────────────────

if "latest_data" not in st.session_state:
    st.session_state.latest_data = None
if "count_history" not in st.session_state:
    st.session_state.count_history = []
if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "connected" not in st.session_state:
    st.session_state.connected = False

# ── Sidebar ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🏙️ CitySight")
    st.markdown("**Smart City Vision Analytics**")
    st.divider()

    st.metric("Backend", "🟢 Connected" if st.session_state.connected else "🔴 Disconnected")

    if st.button("🔄 Refresh Connection", use_container_width=True):
        st.session_state.latest_data = None
        st.rerun()

    st.divider()
    st.markdown("### 📡 Live Stats")
    data = st.session_state.latest_data
    if data:
        counts = data.get("counts", {})
        st.metric("Vehicles", sum(v for k, v in counts.items() if k != "person"))
        st.metric("Pedestrians", counts.get("person", 0))
        st.metric("FPS", f"{data.get('fps', 0):.1f}")
        density = data.get("density", "low")
        density_icon = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(density, "⚪")
        st.metric("Density", f"{density_icon} {density.upper()}")
    else:
        st.caption("Waiting for data...")

    st.divider()
    st.markdown("### 🔔 Recent Alerts")
    for alert in st.session_state.alerts[-5:]:
        st.warning(alert)

    st.divider()
    st.caption("CitySight v1.0.0 | [Docs](https://github.com)")

# ── Main content ────────────────────────────────────────────────────────

col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("## 📹 Live Feed")
    video_placeholder = st.empty()

with col2:
    st.markdown("## 📊 Vehicle Breakdown")
    chart_placeholder = st.empty()

# ── Row 2: timeline + heatmap ──────────────────────────────────────────

col3, col4 = st.columns(2)

with col3:
    st.markdown("## 📈 Counts Over Time")
    timeline_placeholder = st.empty()

with col4:
    st.markdown("## 🔥 Activity Heatmap")
    heatmap_placeholder = st.empty()

# ── Row 3: events table ────────────────────────────────────────────────

st.divider()
st.markdown("## 📋 Recent Events")
events_placeholder = st.empty()


# ── Background WebSocket loop ───────────────────────────────────────────

async def websocket_loop():
    """Continuously connect to the backend WS and update session state."""
    while True:
        try:
            async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=10) as ws:
                st.session_state.connected = True
                logger.info("WebSocket connected")
                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    st.session_state.latest_data = data

                    # Track count history
                    counts = data.get("counts", {})
                    ts = time.time()
                    vehicles = sum(v for k, v in counts.items() if k != "person")
                    pedestrians = counts.get("person", 0)
                    st.session_state.count_history.append({
                        "timestamp": ts,
                        "vehicles": vehicles,
                        "pedestrians": pedestrians,
                    })
                    # Keep last 300 points
                    if len(st.session_state.count_history) > 300:
                        st.session_state.count_history = st.session_state.count_history[-300:]

                    # Track alerts
                    alert = data.get("alert")
                    if alert:
                        st.session_state.alerts.append(f"[{datetime.now().strftime('%H:%M:%S')}] {alert}")
                        if len(st.session_state.alerts) > 50:
                            st.session_state.alerts = st.session_state.alerts[-50:]

        except (websockets.ConnectionClosed, OSError, ConnectionRefusedError) as exc:
            st.session_state.connected = False
            logger.warning("WebSocket disconnected: %s — retrying in 2s", exc)
            await asyncio.sleep(2)
        except Exception as exc:
            st.session_state.connected = False
            logger.error("WebSocket error: %s", exc)
            await asyncio.sleep(2)


async def fetch_events() -> list[dict]:
    """Fetch recent events from the backend."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{BACKEND_URL}/events", params={"limit": 20})
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        pass
    return []


def run_ws_loop():
    """Sync wrapper to run the async WebSocket loop in Streamlit."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop.run_until_complete(ws_async_main())


async def ws_async_main():
    """Run the WS loop and refresh the UI."""
    ws_task = asyncio.create_task(websocket_loop())
    events_task = asyncio.create_task(fetch_events())

    # Let them run briefly
    await asyncio.sleep(0.2)
    events_data = await events_task

    # Update UI
    update_ui(events_data)

    # Don't cancel ws_task — Streamlit will re-run the script


def update_ui(events_data: list[dict]):
    """Update all UI components with latest session state."""
    data = st.session_state.latest_data

    # ── Video frame ──────────────────────────────────────────────────
    if data and data.get("frame_b64"):
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

    # ── Vehicle breakdown chart ──────────────────────────────────────
    if data:
        counts = data.get("counts", {})
        vehicle_data = {k: v for k, v in counts.items() if k != "person"}
        if vehicle_data:
            fig = go.Figure(data=[
                go.Bar(
                    x=list(vehicle_data.keys()),
                    y=list(vehicle_data.values()),
                    marker_color=["#FF6B35", "#FFD23F", "#3BCEAC", "#5D9CEC", "#9B5DE5"],
                    text=list(vehicle_data.values()),
                    textposition="auto",
                )
            ])
            fig.update_layout(
                margin=dict(l=20, r=20, t=10, b=20),
                height=280,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ccc"),
                xaxis=dict(title=None),
                yaxis=dict(title=None),
            )
            chart_placeholder.plotly_chart(fig, use_container_width=True)
        else:
            chart_placeholder.info("No vehicles detected")
    else:
        chart_placeholder.info("Waiting for data...")

    # ── Timeline chart ───────────────────────────────────────────────
    hist = st.session_state.count_history
    if hist:
        df = pd.DataFrame(hist)
        df["time"] = pd.to_datetime(df["timestamp"], unit="s")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df["time"], y=df["vehicles"],
            mode="lines", name="Vehicles",
            line=dict(color="#FF6B35", width=2),
        ))
        fig2.add_trace(go.Scatter(
            x=df["time"], y=df["pedestrians"],
            mode="lines", name="Pedestrians",
            line=dict(color="#3BCEAC", width=2),
        ))
        fig2.update_layout(
            margin=dict(l=20, r=20, t=10, b=20),
            height=280,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ccc"),
            legend=dict(orientation="h", yanchor="top", y=1.1),
            xaxis=dict(title=None),
            yaxis=dict(title=None),
        )
        timeline_placeholder.plotly_chart(fig2, use_container_width=True)
    else:
        timeline_placeholder.info("Accumulating data...")

    # ── Heatmap ──────────────────────────────────────────────────────
    if data and data.get("heatmap_b64"):
        heatmap_html = f"""
        <div style="border-radius:8px; overflow:hidden;">
            <img src="data:image/png;base64,{data['heatmap_b64']}"
                 style="width:100%;" />
        </div>
        """
        heatmap_placeholder.markdown(heatmap_html, unsafe_allow_html=True)
    else:
        heatmap_placeholder.info("Heatmap accumulating...")

    # ── Events table ─────────────────────────────────────────────────
    if events_data:
        df_events = pd.DataFrame(events_data)
        cols = ["timestamp", "event_type", "vehicle_count", "pedestrian_count", "density_level", "alert_message"]
        display_cols = [c for c in cols if c in df_events.columns]
        df_display = df_events[display_cols].copy()
        df_display.columns = ["Time", "Type", "Vehicles", "Pedestrians", "Density", "Alert"]
        events_placeholder.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
        )
    else:
        events_placeholder.caption("No events logged yet (configure DATABASE_URL for persistence)")


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_ws_loop()
