from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass(frozen=True)
class TelemetrySnapshot:
    drone_id: str
    timestamp: float
    latitude: float
    longitude: float
    altitude_m: float
    heading_deg: float
    battery_percent: Optional[float] = None

    def age_seconds(self, now: Optional[float] = None) -> float:
        current_time = now if now is not None else datetime.now(timezone.utc).timestamp()
        return max(0.0, current_time - self.timestamp)


@dataclass(frozen=True)
class VisionEvent:
    drone_id: str
    timestamp: float
    anomaly_type: str
    confidence: float
    frame_ref: Optional[str] = None


@dataclass(frozen=True)
class CommanderCommand:
    command: str
    target_system: int
    target_component: int
    x: float
    y: float
    z: float
    reasoning: str

    def as_dict(self) -> dict:
        return {
            "command": self.command,
            "target_system": self.target_system,
            "target_component": self.target_component,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "reasoning": self.reasoning,
        }
