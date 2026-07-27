"""Tests for WeatherPolicy hard-rejection rules in core/weather_policy.py."""
import unittest

from src.core.weather_policy import WeatherPolicyError, validate_weather_for_flight
from src.core.weather_types import WeatherSnapshot


def _make_snapshot(**overrides) -> WeatherSnapshot:
    """Return a clearly-safe weather snapshot unless overridden."""
    base = dict(
        timestamp=1_000_000.0,
        latitude=-35.36,
        longitude=149.16,
        wind_speed_mps=3.0,         # well under 15 m/s
        wind_direction_deg=180.0,
        gust_speed_mps=5.0,         # well under 20 m/s
        temperature_c=22.0,
        precipitation_mm=0.0,       # well under 10 mm
        visibility_m=10000.0,       # well above 1000 m
        lightning_risk=0.0,
        thunderstorm_risk=0.0,
        source="test",
        confidence=1.0,
    )
    base.update(overrides)
    return WeatherSnapshot(**base)


class WeatherPolicyHardRejectTests(unittest.TestCase):
    def test_safe_weather_is_accepted(self):
        """Completely safe conditions must not raise."""
        validate_weather_for_flight(_make_snapshot())  # must not raise

    # --- Lightning / Thunderstorm ---
    def test_rejects_high_lightning_risk(self):
        with self.assertRaises(WeatherPolicyError) as ctx:
            validate_weather_for_flight(_make_snapshot(lightning_risk=0.6))
        self.assertIn("lightning", str(ctx.exception).lower())

    def test_rejects_high_thunderstorm_risk(self):
        with self.assertRaises(WeatherPolicyError):
            validate_weather_for_flight(_make_snapshot(thunderstorm_risk=0.51))

    def test_accepts_borderline_low_lightning(self):
        """Risk exactly at 0.5 is the boundary — must NOT be rejected."""
        validate_weather_for_flight(_make_snapshot(lightning_risk=0.5))  # must not raise

    # --- Wind Speed ---
    def test_rejects_excessive_wind(self):
        with self.assertRaises(WeatherPolicyError) as ctx:
            validate_weather_for_flight(_make_snapshot(wind_speed_mps=15.1))
        self.assertIn("wind speed", str(ctx.exception).lower())

    def test_accepts_wind_at_limit(self):
        validate_weather_for_flight(_make_snapshot(wind_speed_mps=15.0))  # must not raise

    def test_accepts_calm_wind(self):
        validate_weather_for_flight(_make_snapshot(wind_speed_mps=0.0))  # must not raise

    # --- Gust Speed ---
    def test_rejects_excessive_gusts(self):
        with self.assertRaises(WeatherPolicyError) as ctx:
            validate_weather_for_flight(_make_snapshot(gust_speed_mps=20.1))
        self.assertIn("gust", str(ctx.exception).lower())

    def test_accepts_gusts_at_limit(self):
        validate_weather_for_flight(_make_snapshot(gust_speed_mps=20.0))  # must not raise

    # --- Visibility ---
    def test_rejects_low_visibility(self):
        with self.assertRaises(WeatherPolicyError) as ctx:
            validate_weather_for_flight(_make_snapshot(visibility_m=999.0))
        self.assertIn("visibility", str(ctx.exception).lower())

    def test_accepts_visibility_at_limit(self):
        validate_weather_for_flight(_make_snapshot(visibility_m=1000.0))  # must not raise

    def test_rejects_zero_visibility(self):
        with self.assertRaises(WeatherPolicyError):
            validate_weather_for_flight(_make_snapshot(visibility_m=0.0))

    # --- Precipitation ---
    def test_rejects_heavy_precipitation(self):
        with self.assertRaises(WeatherPolicyError) as ctx:
            validate_weather_for_flight(_make_snapshot(precipitation_mm=10.1))
        self.assertIn("precipitation", str(ctx.exception).lower())

    def test_accepts_precipitation_at_limit(self):
        validate_weather_for_flight(_make_snapshot(precipitation_mm=10.0))  # must not raise

    def test_accepts_light_precipitation(self):
        validate_weather_for_flight(_make_snapshot(precipitation_mm=2.0))  # must not raise

    # --- Compound Cases ---
    def test_rejects_on_first_violation(self):
        """When multiple conditions fail, an exception must still be raised."""
        with self.assertRaises(WeatherPolicyError):
            validate_weather_for_flight(_make_snapshot(
                lightning_risk=1.0,
                wind_speed_mps=30.0,
                visibility_m=0.0,
            ))


if __name__ == "__main__":
    unittest.main()
