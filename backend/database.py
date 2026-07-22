"""Database layer — Neon PostgreSQL for event logging.

Uses asyncpg for async access. Falls back gracefully to no-op when
DATABASE_URL is not configured (everything else still works).
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

from backend.config import config

logger = logging.getLogger("citysight.db")

_initialised = False
_pool = None

INIT_SQL = """
CREATE TABLE IF NOT EXISTS events (
    id          BIGSERIAL PRIMARY KEY,
    timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type  TEXT NOT NULL,
    vehicle_count INTEGER NOT NULL DEFAULT 0,
    pedestrian_count INTEGER NOT NULL DEFAULT 0,
    density_level TEXT NOT NULL DEFAULT 'low',
    alert_message TEXT,
    details     JSONB
);

CREATE INDEX IF NOT EXISTS idx_events_ts ON events (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_events_type ON events (event_type);
"""


async def init_db() -> bool:
    """Initialise the connection pool and run schema migration."""
    global _initialised, _pool
    if not config.database_url:
        logger.info("DATABASE_URL not set — event logging disabled")
        return False
    try:
        import asyncpg
        _pool = await asyncpg.create_pool(
            config.database_url,
            min_size=1,
            max_size=4,
            statement_cache_size=0,
        )
        async with _pool.acquire() as conn:
            await conn.execute(INIT_SQL)
        _initialised = True
        logger.info("Connected to Neon PostgreSQL")
        return True
    except ImportError:
        logger.warning("asyncpg not installed; database logging disabled")
        return False
    except Exception as exc:
        logger.error("Failed to connect to database: %s", exc)
        return False


async def close_db() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


async def log_event(
    event_type: str,
    vehicle_count: int,
    pedestrian_count: int,
    density_level: str,
    alert_message: str | None = None,
    details: dict | None = None,
) -> bool:
    """Insert an event row. Returns True on success."""
    if not _initialised or _pool is None:
        return False
    try:
        async with _pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO events
                   (event_type, vehicle_count, pedestrian_count, density_level, alert_message, details)
                   VALUES ($1, $2, $3, $4, $5, $6)""",
                event_type,
                vehicle_count,
                pedestrian_count,
                density_level,
                alert_message,
                json.dumps(details) if details else None,
            )
        return True
    except Exception as exc:
        logger.error("Failed to log event: %s", exc)
        return False


async def recent_events(limit: int = 20) -> list[dict]:
    """Return the most recent events for the dashboard."""
    if not _initialised or _pool is None:
        return []
    try:
        async with _pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM events ORDER BY timestamp DESC LIMIT $1", limit
            )
            return [dict(r) for r in rows]
    except Exception as exc:
        logger.error("Failed to fetch events: %s", exc)
        return []


async def hourly_stats(hours: int = 24) -> list[dict]:
    """Return hourly aggregated counts for charts."""
    if not _initialised or _pool is None:
        return []
    try:
        async with _pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    date_trunc('hour', timestamp) AS hour,
                    AVG(vehicle_count)::int AS avg_vehicles,
                    AVG(pedestrian_count)::int AS avg_pedestrians,
                    COUNT(*) AS event_count
                FROM events
                WHERE timestamp > NOW() - ($1 || ' hours')::interval
                GROUP BY hour
                ORDER BY hour
                """,
                str(hours),
            )
            return [dict(r) for r in rows]
    except Exception as exc:
        logger.error("Failed to fetch stats: %s", exc)
        return []
