import time
import requests
import asyncio
from typing import Optional
from .weather_types import WeatherSnapshot

class WeatherProvider:
    def __init__(self, use_fake: bool = False):
        self.use_fake = use_fake
        self.base_url = "https://api.open-meteo.com/v1/forecast"

    async def get_weather(self, lat: float, lon: float) -> Optional[WeatherSnapshot]:
        if self.use_fake:
            return self._get_fake_weather(lat, lon)
        
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._fetch_open_meteo, lat, lon)

    def _fetch_open_meteo(self, lat: float, lon: float) -> Optional[WeatherSnapshot]:
        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,wind_speed_10m,wind_direction_10m,wind_gusts_10m,precipitation,weather_code",
            }
            # Add timeout for safety
            response = requests.get(self.base_url, params=params, timeout=5.0)
            response.raise_for_status()
            data = response.json()

            current = data.get("current", {})
            
            # Simple heuristic for lightning/thunderstorm based on WMO weather codes (95, 96, 99)
            code = current.get("weather_code", 0)
            thunderstorm_risk = 1.0 if code in [95, 96, 99] else 0.0
            
            # Visibility is not always in current for open-meteo basic tier, default to 10000m (clear)
            visibility_m = 10000.0

            return WeatherSnapshot(
                timestamp=time.time(),
                latitude=lat,
                longitude=lon,
                wind_speed_mps=current.get("wind_speed_10m", 0.0) / 3.6, # convert km/h to m/s
                wind_direction_deg=current.get("wind_direction_10m", 0.0),
                gust_speed_mps=current.get("wind_gusts_10m", 0.0) / 3.6, # convert km/h to m/s
                temperature_c=current.get("temperature_2m", 0.0),
                precipitation_mm=current.get("precipitation", 0.0),
                visibility_m=visibility_m,
                lightning_risk=thunderstorm_risk,
                thunderstorm_risk=thunderstorm_risk,
                source="open-meteo",
                confidence=0.9
            )
        except Exception as e:
            print(f"[WeatherProvider] Error fetching weather: {e}")
            return None

    def _get_fake_weather(self, lat: float, lon: float) -> WeatherSnapshot:
        return WeatherSnapshot(
            timestamp=time.time(),
            latitude=lat,
            longitude=lon,
            wind_speed_mps=5.0,
            wind_direction_deg=180.0,
            gust_speed_mps=8.0,
            temperature_c=22.0,
            precipitation_mm=0.0,
            visibility_m=10000.0,
            lightning_risk=0.0,
            thunderstorm_risk=0.0,
            source="fake",
            confidence=1.0
        )
