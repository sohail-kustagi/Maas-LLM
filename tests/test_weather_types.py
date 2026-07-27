"""Tests for WeatherSnapshot and WeatherAlert in core/weather_types.py."""
import time
import unittest

from src.core.weather_types import WeatherAlert, WeatherSnapshot


def _make_snapshot(**overrides) -> WeatherSnapshot:
    base = dict(
        timestamp=1_000_000.0,
        latitude=-35.36,
        longitude=149.16,
        wind_speed_mps=5.0,
        wind_direction_deg=180.0,
        gust_speed_mps=8.0,
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


class WeatherSnapshotTests(unittest.TestCase):
    def test_all_fields_stored(self):
        snap = _make_snapshot()
        self.assertAlmostEqual(snap.wind_speed_mps, 5.0)
        self.assertEqual(snap.source, "test")
        self.assertEqual(snap.confidence, 1.0)
        self.assertEqual(snap.lightning_risk, 0.0)

    def test_age_seconds_positive(self):
        snap = _make_snapshot(timestamp=1_000_000.0)
        self.assertAlmostEqual(snap.age_seconds(1_000_010.0), 10.0)

    def test_age_seconds_clamps_negative(self):
        snap = _make_snapshot(timestamp=1_000_010.0)
        # now is earlier — must clamp to 0
        self.assertEqual(snap.age_seconds(1_000_000.0), 0.0)

    def test_thunderstorm_risk_boundary(self):
        clear = _make_snapshot(thunderstorm_risk=0.0)
        stormy = _make_snapshot(thunderstorm_risk=1.0)
        self.assertEqual(clear.thunderstorm_risk, 0.0)
        self.assertEqual(stormy.thunderstorm_risk, 1.0)

    def test_snapshot_is_frozen(self):
        snap = _make_snapshot()
        with self.assertRaises((AttributeError, TypeError)):
            snap.wind_speed_mps = 999.0  # type: ignore[misc]

    def test_high_wind_values(self):
        snap = _make_snapshot(wind_speed_mps=25.0, gust_speed_mps=35.0)
        self.assertGreater(snap.wind_speed_mps, 15.0)
        self.assertGreater(snap.gust_speed_mps, 20.0)

    def test_zero_visibility(self):
        snap = _make_snapshot(visibility_m=0.0)
        self.assertEqual(snap.visibility_m, 0.0)

    def test_heavy_precipitation(self):
        snap = _make_snapshot(precipitation_mm=50.0)
        self.assertEqual(snap.precipitation_mm, 50.0)


class WeatherAlertTests(unittest.TestCase):
    def test_alert_fields(self):
        alert = WeatherAlert(
            alert_type="HIGH_WIND",
            severity="WARNING",
            message="Wind speed 18 m/s detected.",
            timestamp=1_000_000.0,
            latitude=-35.36,
            longitude=149.16,
        )
        self.assertEqual(alert.alert_type, "HIGH_WIND")
        self.assertEqual(alert.severity, "WARNING")
        self.assertIn("18", alert.message)

    def test_critical_alert(self):
        alert = WeatherAlert(
            alert_type="LIGHTNING_RISK",
            severity="CRITICAL",
            message="Lightning detected nearby.",
            timestamp=0.0,
            latitude=0.0,
            longitude=0.0,
        )
        self.assertEqual(alert.severity, "CRITICAL")
        self.assertEqual(alert.alert_type, "LIGHTNING_RISK")

    def test_all_defined_alert_types(self):
        """Ensure all alert types from the plan can be instantiated."""
        known_types = [
            "HIGH_WIND", "HIGH_GUSTS", "LOW_VISIBILITY", "HEAVY_PRECIPITATION",
            "THUNDERSTORM_RISK", "LIGHTNING_RISK",
        ]
        for alert_type in known_types:
            alert = WeatherAlert(
                alert_type=alert_type,
                severity="WARNING",
                message=f"Test {alert_type}",
                timestamp=0.0, latitude=0.0, longitude=0.0,
            )
            self.assertEqual(alert.alert_type, alert_type)


if __name__ == "__main__":
    unittest.main()
