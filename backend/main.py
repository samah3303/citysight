"""CitySight — FastAPI + WebSocket server.

Routes:
  GET  /              — health check + status
  GET  /state         — current shared state (JSON)
  GET  /events        — recent DB events
  GET  /stats/hourly  — hourly aggregated stats
  WS   /ws            — real-time frame + analytics stream
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.config import config
from backend.streamer import Streamer, clients, shared_state, state_lock
from backend.database import init_db, close_db, recent_events, hourly_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("citysight")

app = FastAPI(title="CitySight", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

streamer: Streamer | None = None


# ── Lifecycle ──────────────────────────────────────────────────────────────

@app.on_event("startup")
async def on_startup():
    global streamer
    await init_db()
    streamer = Streamer()
    await streamer.start()


@app.on_event("shutdown")
async def on_shutdown():
    if streamer:
        await streamer.stop()
    await close_db()


# ── HTTP endpoints ─────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": "CitySight",
        "version": "1.0.0",
        "status": "running" if streamer and streamer._running else "stopped",
        "model": config.model_name,
        "video_source": config.video_source,
        "fps": shared_state.get("fps", 0),
    }


@app.get("/state")
async def get_state():
    async with state_lock:
        return {
            "counts": shared_state.get("counts", {}),
            "density": shared_state.get("density", "low"),
            "fps": shared_state.get("fps", 0),
            "alert": shared_state.get("alert"),
            "object_count": len(shared_state.get("objects", [])),
        }


@app.get("/events")
async def get_events(limit: int = 20):
    events = await recent_events(limit)
    # Convert datetime to string for JSON serialisation
    for e in events:
        for key in ("timestamp", "hour"):
            if key in e and e[key] is not None:
                e[key] = e[key].isoformat()
    return events


@app.get("/stats/hourly")
async def get_hourly_stats(hours: int = 24):
    stats = await hourly_stats(hours)
    for s in stats:
        if "hour" in s and s["hour"] is not None:
            s["hour"] = s["hour"].isoformat()
    return stats


# ── WebSocket ──────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    logger.info("WebSocket client connected (total: %d)", len(clients))
    try:
        # Keep the connection alive; the streamer pushes data
        while True:
            # Read incoming (pings / close)
            data = await ws.receive_text()
            # Echo back any message as a heartbeat
            if data == "ping":
                await ws.send_text("pong")
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        clients.discard(ws)
        logger.info("WebSocket client disconnected (total: %d)", len(clients))


# ── Entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=config.host,
        port=config.port,
        reload=False,
        log_level="info",
    )
