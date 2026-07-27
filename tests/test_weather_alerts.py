"""Tests for weather alert generation in core/weather_alerts.py."""
import unittest

from src.core.weather_alerts import generate_weather_alerts
from src.core.weather_types import WeatherSnapshot


def _make_snapshot(**overrides) -> WeatherSnapshot:
    """Return a clear-sky snapshot unless overridden."""
    base = dict(
        timestamp=1_000_000.0,
        latitude=-35.36,
        longitude=149.16,
        wind_speed_mps=3.0,
        wind_direction_deg=180.0,
        gust_speed_mps=5.0,
        temperature_c=22.0,
        precipitation_mm=0.0,
        visibility_m=10000.0,
        lightning_risk=0.0,
        thunderstorm_risk=0.0,
        source="test",
        confidence=1.0,
    )
    base.update(overrides)
    return WeatherSnapshot(**base)


def _alert_types(snap: WeatherSnapshot) -> list[str]:
    return [a.alert_type for a in generate_weather_alerts(snap)]


class WeatherAlertsTests(unittest.TestCase):
    def test_clear_weather_produces_no_alerts(self):
        alerts = generate_weather_alerts(_make_snapshot())
        self.assertEqual(alerts, [])

    # --- HIGH_WIND ---
    def test_no_high_wind_alert_below_threshold(self):
        self.assertNotIn("HIGH_WIND", _alert_types(_make_snapshot(wind_speed_mps=9.9)))

    def test_high_wind_alert_at_threshold(self):
        """Wind at exactly 10 m/s should trigger HIGH_WIND."""
        self.assertIn("HIGH_WIND", _alert_types(_make_snapshot(wind_speed_mps=10.0)))

    def test_high_wind_alert_above_threshold(self):
        self.assertIn("HIGH_WIND", _alert_types(_make_snapshot(wind_speed_mps=14.0)))

    def test_high_wind_alert_has_warning_severity(self):
        alerts = generate_weather_alerts(_make_snapshot(wind_speed_mps=12.0))
        wind_alerts = [a for a in alerts if a.alert_type == "HIGH_WIND"]
        self.assertEqual(len(wind_alerts), 1)
        self.assertEqual(wind_alerts[0].severity, "WARNING")

    # --- HIGH_GUSTS ---
    def test_no_gust_alert_below_threshold(self):
        self.assertNotIn("HIGH_GUSTS", _alert_types(_make_snapshot(gust_speed_mps=14.9)))

    def test_high_gusts_alert_at_threshold(self):
        self.assertIn("HIGH_GUSTS", _alert_types(_make_snapshot(gust_speed_mps=15.0)))

    # --- LOW_VISIBILITY ---
    def test_no_visibility_alert_above_threshold(self):
        self.assertNotIn("LOW_VISIBILITY", _alert_types(_make_snapshot(visibility_m=3001.0)))

    def test_low_visibility_alert_at_threshold(self):
        self.assertIn("LOW_VISIBILITY", _alert_types(_make_snapshot(visibility_m=3000.0)))

    def test_low_visibility_alert_below_threshold(self):
        self.assertIn("LOW_VISIBILITY", _alert_types(_make_snapshot(visibility_m=500.0)))

    # --- HEAVY_PRECIPITATION ---
    def test_no_precip_alert_below_threshold(self):
        self.assertNotIn("HEAVY_PRECIPITATION", _alert_types(_make_snapshot(precipitation_mm=1.9)))

    def test_heavy_precip_alert_at_threshold(self):
        self.assertIn("HEAVY_PRECIPITATION", _alert_types(_make_snapshot(precipitation_mm=2.0)))

    # --- Multiple Simultaneous Alerts ---
    def test_multiple_alerts_in_bad_conditions(self):
        alert_types = _alert_types(_make_snapshot(
            wind_speed_mps=12.0,
            gust_speed_mps=18.0,
            visibility_m=500.0,
            precipitation_mm=5.0,
        ))
        self.assertIn("HIGH_WIND", alert_types)
        self.assertIn("HIGH_GUSTS", alert_types)
        self.assertIn("LOW_VISIBILITY", alert_types)
        self.assertIn("HEAVY_PRECIPITATION", alert_types)

    # --- Alert metadata correctness ---
    def test_alert_has_correct_location(self):
        alerts = generate_weather_alerts(_make_snapshot(
            latitude=12.34, longitude=56.78, wind_speed_mps=11.0
        ))
        wind_alert = next(a for a in alerts if a.alert_type == "HIGH_WIND")
        self.assertAlmostEqual(wind_alert.latitude, 12.34)
        self.assertAlmostEqual(wind_alert.longitude, 56.78)

    def test_alert_message_contains_value(self):
        alerts = generate_weather_alerts(_make_snapshot(wind_speed_mps=11.0))
        wind_alert = next(a for a in alerts if a.alert_type == "HIGH_WIND")
        self.assertIn("11.0", wind_alert.message)


if __name__ == "__main__":
    unittest.main()
