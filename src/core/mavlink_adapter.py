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

    # ArduPilot custom_mode -> flight mode name mapping (most common modes)
    _FLIGHT_MODE_MAP = {
        0: "STABILIZE", 2: "ALT_HOLD", 3: "AUTO", 4: "GUIDED",
        5: "LOITER", 6: "RTL", 9: "LAND", 16: "POSHOLD",
    }

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
            # Decode armed state from base_mode (bit 7 = MAV_MODE_FLAG_SAFETY_ARMED)
            base_mode = getattr(message, "base_mode", 0)
            self.values["is_armed"] = bool(base_mode & 0x80)
            # Decode flight mode from custom_mode
            custom_mode = getattr(message, "custom_mode", None)
            if custom_mode is not None:
                self.values["flight_mode"] = self._FLIGHT_MODE_MAP.get(custom_mode, f"MODE_{custom_mode}")
        elif message_type == "VFR_HUD":
            self.values["ground_speed_mps"] = float(getattr(message, "groundspeed", 0.0))
            self.values["climb_rate_mps"] = float(getattr(message, "climb", 0.0))
        elif message_type == "GPS_RAW_INT":
            self.values["gps_fix_type"] = int(getattr(message, "fix_type", 0))
            self.values["satellites_visible"] = int(getattr(message, "satellites_visible", 0))
        elif message_type == "HOME_POSITION":
            self.values["home_latitude"] = getattr(message, "latitude", 0) / 10_000_000.0
            self.values["home_longitude"] = getattr(message, "longitude", 0) / 10_000_000.0
            self.values["home_altitude_m"] = getattr(message, "altitude", 0) / 1000.0

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
            ground_speed_mps=self.values.get("ground_speed_mps"),
            climb_rate_mps=self.values.get("climb_rate_mps"),
            gps_fix_type=self.values.get("gps_fix_type"),
            satellites_visible=self.values.get("satellites_visible"),
            flight_mode=self.values.get("flight_mode"),
            is_armed=self.values.get("is_armed"),
            home_latitude=self.values.get("home_latitude"),
            home_longitude=self.values.get("home_longitude"),
            home_altitude_m=self.values.get("home_altitude_m"),
        )
        
    @staticmethod
    def build_set_position_target_global_int(lat: float, lon: float, alt_m: float, mav):
        """Builds a MAVLink SET_POSITION_TARGET_GLOBAL_INT message for absolute GPS navigation."""
        # coordinate_frame 6 = MAV_FRAME_GLOBAL_RELATIVE_ALT_INT
        # type_mask 0x0DF8 = ignore velocity/accel/yaw, only use position
        lat_int = int(lat * 10_000_000)
        lon_int = int(lon * 10_000_000)
        
        return mav.mav.set_position_target_global_int_encode(
            0,       # time_boot_ms
            1,       # target_system
            1,       # target_component
            6,       # coordinate_frame
            0x0DF8,  # type_mask
            lat_int, # lat_int
            lon_int, # lon_int
            alt_m,   # alt
            0, 0, 0, # vx, vy, vz
            0, 0, 0, # afx, afy, afz
            0, 0     # yaw, yaw_rate
        )
