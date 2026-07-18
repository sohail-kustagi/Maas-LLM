import time
from typing import Any, Iterable, Optional

from .types import TelemetrySnapshot


class MAVLinkDependencyError(RuntimeError):
    """Raised when raw MAVLink decoding is requested without pymavlink."""


class MAVLinkTelemetryDecoder:
    """Decode common ArduPilot telemetry messages into a snapshot."""

    def __init__(self):
        try:
            from pymavlink.dialects.v20 import common as mavlink
        except ImportError as error:
            raise MAVLinkDependencyError(
                "Install pymavlink to decode raw SITL MAVLink packets"
            ) from error
        self.parser = mavlink.MAVLink(None)
        self.values = {}

    def feed(self, data: bytes, received_at: Optional[float] = None) -> Optional[TelemetrySnapshot]:
        messages = self.parser.parse_buffer(data) or []
        for message in messages:
            self._update(message)
        return self.snapshot(received_at)

    def _update(self, message: Any) -> None:
        message_type = message.get_type()
        if message_type == "GLOBAL_POSITION_INT":
            self.values.update(
                latitude=message.lat / 10_000_000.0,
                longitude=message.lon / 10_000_000.0,
                altitude_m=message.relative_alt / 1000.0,
                heading_deg=(message.hdg / 100.0 if message.hdg != 65535 else None),
            )
        elif message_type == "SYS_STATUS":
            if message.battery_remaining >= 0:
                self.values["battery_percent"] = float(message.battery_remaining)
        elif message_type == "HEARTBEAT":
            self.values["drone_id"] = f"mav-{message.get_srcSystem()}"

    def snapshot(self, received_at: Optional[float] = None) -> Optional[TelemetrySnapshot]:
        required = {"drone_id", "latitude", "longitude", "altitude_m", "heading_deg"}
        if not required.issubset(self.values):
            return None
        return TelemetrySnapshot(
            drone_id=self.values["drone_id"],
            timestamp=received_at if received_at is not None else time.time(),
            latitude=self.values["latitude"],
            longitude=self.values["longitude"],
            altitude_m=self.values["altitude_m"],
            heading_deg=self.values["heading_deg"],
            battery_percent=self.values.get("battery_percent"),
        )
