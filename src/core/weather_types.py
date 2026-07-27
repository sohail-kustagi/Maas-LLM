from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class WeatherSnapshot:
    timestamp: float
    latitude: float
    longitude: float
    wind_speed_mps: float
    wind_direction_deg: float
    gust_speed_mps: float
    temperature_c: float
    precipitation_mm: float
    visibility_m: float
    lightning_risk: float  # 0.0 to 1.0 (probability or index)
    thunderstorm_risk: float # 0.0 to 1.0
    source: str
    confidence: float

    def age_seconds(self, now: float) -> float:
        return max(0.0, now - self.timestamp)

@dataclass(frozen=True)
class WeatherAlert:
    alert_type: str  # e.g., "HIGH_WIND", "LOW_VISIBILITY", "LIGHTNING_RISK"
    severity: str    # "WARNING", "CRITICAL"
    message: str
    timestamp: float
    latitude: float
    longitude: float
