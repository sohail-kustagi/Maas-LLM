from typing import List, Optional
from .weather_types import WeatherSnapshot

# Weather thresholds for drone safety
MAX_WIND_SPEED_MPS = 15.0
MAX_GUST_SPEED_MPS = 20.0
MIN_VISIBILITY_M = 1000.0
MAX_PRECIPITATION_MM = 10.0

class WeatherPolicyError(ValueError):
    """Raised when weather conditions strictly forbid the flight."""
    pass

def validate_weather_for_flight(weather: WeatherSnapshot) -> None:
    """
    Validates weather against safety thresholds.
    Raises WeatherPolicyError if flight is unsafe.
    """
    if weather.lightning_risk > 0.5 or weather.thunderstorm_risk > 0.5:
        raise WeatherPolicyError("Flight rejected: High thunderstorm or lightning risk detected.")
    
    if weather.wind_speed_mps > MAX_WIND_SPEED_MPS:
        raise WeatherPolicyError(f"Flight rejected: Wind speed {weather.wind_speed_mps:.1f} m/s exceeds safe limit of {MAX_WIND_SPEED_MPS} m/s.")
        
    if weather.gust_speed_mps > MAX_GUST_SPEED_MPS:
        raise WeatherPolicyError(f"Flight rejected: Gusts {weather.gust_speed_mps:.1f} m/s exceed safe limit of {MAX_GUST_SPEED_MPS} m/s.")
        
    if weather.visibility_m < MIN_VISIBILITY_M:
        raise WeatherPolicyError(f"Flight rejected: Visibility {weather.visibility_m:.1f} m is below safe limit of {MIN_VISIBILITY_M} m.")
        
    if weather.precipitation_mm > MAX_PRECIPITATION_MM:
        raise WeatherPolicyError(f"Flight rejected: Precipitation {weather.precipitation_mm:.1f} mm exceeds safe limit of {MAX_PRECIPITATION_MM} mm.")
