"""DeepSeek-powered anomaly alerting.

When metrics deviate significantly from rolling averages, this module
queries DeepSeek to generate a natural-language alert explaining the
situation for city operators.

Falls back gracefully — if the API key is not set, alerting is disabled.
"""

from __future__ import annotations

import logging

from backend.config import config

logger = logging.getLogger("citysight.alerts")


async def check_and_alert(
    vehicle_count: int,
    vehicle_avg: float,
    pedestrian_count: int,
    pedestrian_avg: float,
    density: str,
    vehicle_types: dict[str, int],
) -> str | None:
    """Return an alert string if deviation exceeds threshold, else None."""
    if not config.deepseek_api_key:
        return None

    v_dev = _deviation(vehicle_count, vehicle_avg)
    p_dev = _deviation(pedestrian_count, pedestrian_avg)
    threshold = config.alert_deviation_threshold

    max_dev = max(v_dev, p_dev)
    if max_dev < threshold:
        return None

    prompt = _build_prompt(
        vehicle_count, vehicle_avg, v_dev,
        pedestrian_count, pedestrian_avg, p_dev,
        density, vehicle_types,
    )

    try:
        import httpx
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                "https://api.deepseek.com/chat/completions",
                headers={
                    "Authorization": f"Bearer {config.deepseek_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": config.deepseek_model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 150,
                    "temperature": 0.3,
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            else:
                logger.warning("DeepSeek API error %d: %s", resp.status_code, resp.text[:200])
                return _fallback_alert(
                    vehicle_count, vehicle_avg, pedestrian_count, pedestrian_avg, density
                )
    except Exception as exc:
        logger.error("DeepSeek API call failed: %s", exc)
        return _fallback_alert(
            vehicle_count, vehicle_avg, pedestrian_count, pedestrian_avg, density
        )


# ── Helpers ────────────────────────────────────────────────────────────────

def _deviation(current: int, avg: float) -> float:
    if avg <= 0:
        return 0.0
    return abs(current - avg) / avg


def _build_prompt(
    vc, va, vd, pc, pa, pd, density, vtypes,
) -> str:
    direction_v = "increase" if vc > va else "decrease"
    direction_p = "increase" if pc > pa else "decrease"
    return (
        f"Traffic snapshot:\n"
        f"- Vehicles: {vc} (avg {va:.1f}, {vd:.0%} {direction_v})\n"
        f"- Pedestrians: {pc} (avg {pa:.1f}, {pd:.0%} {direction_p})\n"
        f"- Crowd density: {density}\n"
        f"- Vehicle breakdown: {vtypes}\n\n"
        f"Generate a single-sentence alert for city operators about any anomalies."
    )


def _fallback_alert(vc, va, pc, pa, density) -> str:
    """Rule-based fallback when DeepSeek is unavailable."""
    parts = []
    if va > 0 and abs(vc - va) / va > config.alert_deviation_threshold:
        direction = "increase" if vc > va else "decrease"
        parts.append(f"⚠️ Vehicle count {vc} — {abs(vc - va) / va:.0%} {direction} vs average of {va:.1f}")
    if pa > 0 and abs(pc - pa) / pa > config.alert_deviation_threshold:
        direction = "increase" if pc > pa else "decrease"
        parts.append(f"⚠️ Pedestrian count {pc} — {abs(pc - pa) / pa:.0%} {direction} vs average of {pa:.1f}")
    if density == "high":
        parts.append("🚨 High crowd density detected")
    return " | ".join(parts) if parts else "No significant anomalies detected."


SYSTEM_PROMPT = """You are a smart city traffic monitoring assistant. 
Given traffic and pedestrian counts compared to their averages, generate a 
concise, actionable alert for city operators. Keep it to one sentence. 
Be specific: mention the deviation magnitude and direction. 
Example: "Vehicle count is 23% above the hourly average — possible congestion forming on the main road."
Do not hallucinate extra detail. Be factual."""
