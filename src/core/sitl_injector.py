"""
sitl_injector.py — Scripted VisionEvent generator for SITL testing.

Injects synthetic disaster detection events on a timer so the
Analyst → Commander pipeline can be validated against real SITL telemetry
without requiring a physical camera.

Injection schedule (relative to start):
  t + 10s  → human_survivor (confidence 0.91)
  t + 30s  → fire            (confidence 0.88)
  t + 60s  → flood_water     (confidence 0.85)
  t + 90s  → vehicle         (confidence 0.79)
  t + 120s → STOP (injector exits gracefully)

The injector is intentionally simple — it emits events even if telemetry is
not yet available. The main pipeline's process_watchdog_events already guards
against missing telemetry.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from .types import VisionEvent

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Scenario schedule
# ──────────────────────────────────────────────────────────────────────────────

# (delay_seconds, anomaly_type, confidence)
SCENARIOS = {
    "flood": [
        (10.0,  "flood_water", 0.95),
        (30.0,  "human_survivor", 0.88),
        (60.0,  "boat", 0.85),
        (90.0,  "vehicle", 0.79),
        (120.0, None, 0.0),
    ],
    "search_and_rescue": [
        (10.0,  "human_survivor", 0.91),
        (30.0,  "vehicle", 0.88),
        (60.0,  "human_survivor", 0.85),
        (90.0,  "bicycle", 0.79),
        (120.0, None, 0.0),
    ],
    "fire": [
        (10.0,  "fire", 0.91),
        (30.0,  "fire_hydrant", 0.88),
        (60.0,  "vehicle", 0.85),
        (90.0,  "human_survivor", 0.79),
        (120.0, None, 0.0),
    ],
    "infrastructure": [
        (10.0,  "train", 0.91),
        (30.0,  "truck", 0.88),
        (60.0,  "vehicle", 0.85),
        (90.0,  "stop_sign", 0.79),
        (120.0, None, 0.0),
    ],
    "free": [
        (10.0,  "human_survivor", 0.91),
        (120.0, None, 0.0),
    ],
    "sandbox": [
        (10.0,  "human_survivor", 0.91),
        (30.0,  "fire", 0.88),
        (60.0,  "flood_water", 0.85),
        (90.0,  "vehicle", 0.79),
        (120.0, None, 0.0),
    ]
}


class SITLInjector:
    """Emits VisionEvents on a scripted schedule into an asyncio Queue.

    Usage:
        injector = SITLInjector(event_queue, drone_id="sitl-drone-1")
        await injector.run()  # runs until scenario ends or task is cancelled
    """

    def __init__(
        self,
        event_queue: asyncio.Queue,
        drone_id: str = "sitl-drone-1",
        scenario: Optional[list[tuple[float, str, float]]] = None,
        mission_profile_name: str = "sandbox"
    ) -> None:
        self.event_queue = event_queue
        self.drone_id = drone_id
        self.scenario = scenario or SCENARIOS.get(mission_profile_name, SCENARIOS["sandbox"])
        self._started_at: float = 0.0
        self._injected: list[VisionEvent] = []

    @property
    def injected_count(self) -> int:
        return len(self._injected)

    async def run(self) -> None:
        """Run the injection schedule. Returns when the scenario is complete."""
        self._started_at = time.time()
        logger.info("[SITLInjector] Starting scripted scenario with %d events.", len(self.scenario) - 1)

        for delay, anomaly_type, confidence in self.scenario:
            elapsed = time.time() - self._started_at
            remaining = max(0.0, delay - elapsed)

            if remaining > 0:
                await asyncio.sleep(remaining)

            if anomaly_type is None:
                logger.info("[SITLInjector] Scenario complete. %d events injected.", self.injected_count)
                return

            event = VisionEvent(
                drone_id=self.drone_id,
                timestamp=time.time(),
                anomaly_type=anomaly_type,
                confidence=confidence,
            )
            await self.event_queue.put(event)
            self._injected.append(event)
            logger.info(
                "[SITLInjector] Injected event #%d: %s (conf=%.2f)",
                self.injected_count, anomaly_type, confidence
            )

    async def run_loop(self, repeat: bool = False) -> None:
        """Run the scenario optionally in a loop (for long SITL sessions)."""
        while True:
            await self.run()
            if not repeat:
                break
            logger.info("[SITLInjector] Repeating scenario...")
            self._injected.clear()
