from typing import List
from .weather_types import WeatherSnapshot, WeatherAlert

# Warning thresholds — alerts trigger at OR above/below these values
HIGH_WIND_THRESHOLD_MPS = 10.0
HIGH_GUSTS_THRESHOLD_MPS = 15.0
LOW_VISIBILITY_THRESHOLD_M = 3000.0
HEAVY_PRECIPITATION_THRESHOLD_MM = 2.0


def generate_weather_alerts(weather: WeatherSnapshot) -> List[WeatherAlert]:
    """Generate advisory warnings based on current weather conditions.

    These are non-fatal warnings that inform planning. Hard rejections are
    handled separately by weather_policy.py.
    """
    alerts = []

    if weather.wind_speed_mps >= HIGH_WIND_THRESHOLD_MPS:
        alerts.append(WeatherAlert(
            alert_type="HIGH_WIND",
            severity="WARNING",
            message=f"High wind detected: {weather.wind_speed_mps:.1f} m/s",
            timestamp=weather.timestamp,
            latitude=weather.latitude,
            longitude=weather.longitude
        ))

    if weather.gust_speed_mps >= HIGH_GUSTS_THRESHOLD_MPS:
        alerts.append(WeatherAlert(
            alert_type="HIGH_GUSTS",
            severity="WARNING",
            message=f"High gusts detected: {weather.gust_speed_mps:.1f} m/s",
            timestamp=weather.timestamp,
            latitude=weather.latitude,
            longitude=weather.longitude
        ))

    if weather.visibility_m <= LOW_VISIBILITY_THRESHOLD_M:
        alerts.append(WeatherAlert(
            alert_type="LOW_VISIBILITY",
            severity="WARNING",
            message=f"Low visibility: {weather.visibility_m:.1f} m",
            timestamp=weather.timestamp,
            latitude=weather.latitude,
            longitude=weather.longitude
        ))

    if weather.precipitation_mm >= HEAVY_PRECIPITATION_THRESHOLD_MM:
        alerts.append(WeatherAlert(
            alert_type="HEAVY_PRECIPITATION",
            severity="WARNING",
            message=f"Precipitation detected: {weather.precipitation_mm:.1f} mm",
            timestamp=weather.timestamp,
            latitude=weather.latitude,
            longitude=weather.longitude
        ))

    return alerts
