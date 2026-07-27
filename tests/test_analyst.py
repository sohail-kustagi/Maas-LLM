"""Tests for AnalystNode context generation."""
import unittest

from src.core.route_types import FeasibilityReport
from src.core.types import TelemetrySnapshot, VisionEvent
from src.core.weather_types import WeatherSnapshot


def _make_analyst():
    from src.nodes.analyst import AnalystNode
    return AnalystNode()


def _make_snap(**overrides) -> TelemetrySnapshot:
    import time
    base = dict(
        drone_id="drone-1",
        timestamp=time.time(),
        latitude=-35.363261,
        longitude=149.165230,
        altitude_m=25.0,
        heading_deg=90.0,
        battery_percent=80.0,
    )
    base.update(overrides)
    return TelemetrySnapshot(**base)


def _make_event(anomaly_type: str = "fire", confidence: float = 0.9) -> VisionEvent:
    import time
    return VisionEvent(
        drone_id="drone-1",
        timestamp=time.time(),
        anomaly_type=anomaly_type,
        confidence=confidence,
    )


def _make_weather(**overrides) -> WeatherSnapshot:
    import time
    base = dict(
        timestamp=time.time(),
        latitude=-35.36, longitude=149.16,
        wind_speed_mps=5.0, wind_direction_deg=180.0,
        gust_speed_mps=8.0, temperature_c=22.0,
        precipitation_mm=0.0, visibility_m=10000.0,
        lightning_risk=0.0, thunderstorm_risk=0.0,
        source="test", confidence=1.0,
    )
    base.update(overrides)
    return WeatherSnapshot(**base)


def _make_feasibility(is_feasible=True, risk="LOW") -> FeasibilityReport:
    return FeasibilityReport(
        is_feasible=is_feasible,
        risk_level=risk,
        estimated_distance_m=500.0,
        estimated_duration_s=120.0,
        estimated_battery_percent_used=5.0,
        violations=[] if is_feasible else ["battery low"],
        warnings=[],
        corrective_suggestions=[],
    )


class AnalystContextGenerationTests(unittest.TestCase):
    def setUp(self):
        self.analyst = _make_analyst()

    def test_generate_context_returns_string(self):
        ctx = self.analyst.generate_context("fire", {"lat": -35.36, "lon": 149.16, "alt": 25.0, "heading": 90.0})
        self.assertIsInstance(ctx, str)
        self.assertGreater(len(ctx), 50)

    def test_context_includes_anomaly_type(self):
        for anomaly in ("fire", "human_survivor", "flood_water", "vehicle"):
            ctx = self.analyst.generate_context(anomaly, {"lat": 0.0, "lon": 0.0, "alt": 20.0, "heading": 0.0})
            self.assertIn(anomaly, ctx, f"Anomaly type '{anomaly}' not in context")

    def test_context_includes_telemetry_values(self):
        ctx = self.analyst.generate_context(
            "fire",
            {"lat": -35.363, "lon": 149.165, "alt": 42.7, "heading": 270.0}
        )
        self.assertIn("-35.363", ctx)
        self.assertIn("149.165", ctx)
        self.assertIn("42.7", ctx)
        self.assertIn("270.0", ctx)

    def test_context_without_weather_still_works(self):
        ctx = self.analyst.generate_context("fire", {"lat": 0.0, "lon": 0.0, "alt": 20.0, "heading": 0.0})
        self.assertIsInstance(ctx, str)

    def test_context_includes_weather_data_when_provided(self):
        ctx = self.analyst.generate_context(
            "fire",
            {"lat": 0.0, "lon": 0.0, "alt": 20.0, "heading": 0.0},
            weather={"wind_speed": 12.5, "visibility": 5000.0, "lightning": 0.1}
        )
        self.assertIn("12.5", ctx)
        self.assertIn("5000.0", ctx)

    def test_context_includes_feasibility_when_provided(self):
        ctx = self.analyst.generate_context(
            "fire",
            {"lat": 0.0, "lon": 0.0, "alt": 20.0, "heading": 0.0},
            feasibility={"is_feasible": False, "risk_level": "CRITICAL"}
        )
        self.assertIn("False", ctx)
        self.assertIn("CRITICAL", ctx)

    def test_generate_event_context_uses_snapshot_fields(self):
        snap = _make_snap(latitude=-35.999, longitude=149.111, altitude_m=55.5, heading_deg=180.0)
        event = _make_event("flood_water")
        ctx = self.analyst.generate_event_context(event, snap)
        self.assertIn("flood_water", ctx)
        self.assertIn("-35.999", ctx)
        self.assertIn("149.111", ctx)
        self.assertIn("55.5", ctx)

    def test_generate_event_context_with_weather(self):
        snap = _make_snap()
        event = _make_event("human_survivor")
        weather = _make_weather(wind_speed_mps=9.0, visibility_m=3000.0, lightning_risk=0.0)
        ctx = self.analyst.generate_event_context(event, snap, weather=weather)
        self.assertIn("9.0", ctx)

    def test_generate_event_context_with_feasibility(self):
        snap = _make_snap()
        event = _make_event("fire")
        feasibility = _make_feasibility(is_feasible=False, risk="CRITICAL")
        ctx = self.analyst.generate_event_context(event, snap, feasibility=feasibility)
        self.assertIn("CRITICAL", ctx)

    def test_generate_event_context_with_all_inputs(self):
        snap = _make_snap()
        event = _make_event("vehicle")
        weather = _make_weather()
        feasibility = _make_feasibility()
        ctx = self.analyst.generate_event_context(event, snap, weather=weather, feasibility=feasibility)
        self.assertIsInstance(ctx, str)
        self.assertIn("vehicle", ctx)

    def test_all_anomaly_types_handled(self):
        snap = _make_snap()
        for anomaly in ("human_survivor", "fire", "flood_water", "vehicle", "none"):
            event = _make_event(anomaly)
            ctx = self.analyst.generate_event_context(event, snap)
            self.assertIn(anomaly, ctx)


if __name__ == "__main__":
    unittest.main()
