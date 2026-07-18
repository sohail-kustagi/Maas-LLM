import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Mapping, Optional

from .types import TelemetrySnapshot, VisionEvent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineResult:
    event: VisionEvent
    context: str
    command: Optional[dict]


class PipelineCoordinator:
    """Connect Watchdog, Analyst, and Commander through an async queue."""

    def __init__(
        self,
        analyst: Callable[[VisionEvent, TelemetrySnapshot], str],
        commander: Callable[[str, TelemetrySnapshot], Awaitable[Optional[dict]]],
        telemetry: Callable[[], Optional[TelemetrySnapshot]],
        queue_size: int = 32,
    ):
        self.events: asyncio.Queue[VisionEvent] = asyncio.Queue(maxsize=queue_size)
        self.analyst = analyst
        self.commander = commander
        self.telemetry = telemetry

    async def publish(self, event: VisionEvent) -> None:
        await self.events.put(event)

    async def run_once(self) -> Optional[PipelineResult]:
        event = await self.events.get()
        try:
            telemetry = self.telemetry()
            if telemetry is None:
                logger.warning("Dropping event because telemetry is unavailable")
                return None
            context = self.analyst(event, telemetry)
            command = await self.commander(context, telemetry)
            return PipelineResult(event=event, context=context, command=command)
        finally:
            self.events.task_done()


def parse_sandbox_telemetry(data: bytes) -> TelemetrySnapshot:
    """Parse the explicit local JSON adapter format used by sandbox mode."""
    payload = json.loads(data.decode("utf-8"))
    required = {
        "drone_id",
        "timestamp",
        "lat",
        "lon",
        "alt",
        "heading",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"missing telemetry fields: {', '.join(sorted(missing))}")

    return TelemetrySnapshot(
        drone_id=str(payload["drone_id"]),
        timestamp=float(payload["timestamp"]),
        latitude=float(payload["lat"]),
        longitude=float(payload["lon"]),
        altitude_m=float(payload["alt"]),
        heading_deg=float(payload["heading"]),
        battery_percent=(
            float(payload["battery_percent"])
            if payload.get("battery_percent") is not None
            else None
        ),
    )
