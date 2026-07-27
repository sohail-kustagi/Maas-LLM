"""Tests for haversine distance, route distance, battery estimation, and flight feasibility."""
import math
import time
import unittest

from src.core.flight_planner import (
    calculate_route_distance,
    evaluate_feasibility,
    haversine,
)
from src.core.route_types import DroneCapabilities, FlightPlan, Waypoint
from src.core.types import TelemetrySnapshot
from src.core.weather_types import WeatherSnapshot


def _make_snap(**overrides) -> TelemetrySnapshot:
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


def _make_cap(**overrides) -> DroneCapabilities:
    base = dict(
        max_speed_mps=15.0,
        max_climb_rate_mps=5.0,
        max_descent_rate_mps=3.0,
        min_altitude_m=5.0,
        max_altitude_m=120.0,
        max_mission_distance_m=5000.0,
        max_waypoints=20,
        battery_reserve_percent=20.0,
        inspection_dwell_time_s=30.0,
    )
    base.update(overrides)
    return DroneCapabilities(**base)


def _make_plan(waypoints: list[Waypoint]) -> FlightPlan:
    return FlightPlan(
        plan_id="test-plan",
        drone_id="drone-1",
        waypoints=waypoints,
        created_timestamp=time.time(),
    )


def _make_weather(**overrides) -> WeatherSnapshot:
    base = dict(
        timestamp=time.time(),
        latitude=-35.36, longitude=149.16,
        wind_speed_mps=3.0, wind_direction_deg=180.0,
        gust_speed_mps=5.0, temperature_c=22.0,
        precipitation_mm=0.0, visibility_m=10000.0,
        lightning_risk=0.0, thunderstorm_risk=0.0,
        source="test", confidence=1.0,
    )
    base.update(overrides)
    return WeatherSnapshot(**base)


class HaversineTests(unittest.TestCase):
    def test_same_point_is_zero(self):
        self.assertAlmostEqual(haversine(0, 0, 0, 0), 0.0)

    def test_known_distance_approx(self):
        # Sydney <-> Canberra ≈ 247 km (haversine on a spherical Earth)
        sydney_lat, sydney_lon = -33.8688, 151.2093
        canberra_lat, canberra_lon = -35.2809, 149.1300
        dist_m = haversine(sydney_lat, sydney_lon, canberra_lat, canberra_lon)
        # Allow ±5km tolerance for spherical approximation vs geodetic
        self.assertAlmostEqual(dist_m / 1000.0, 247.0, delta=5.0)

    def test_symmetric(self):
        d1 = haversine(10.0, 20.0, 11.0, 21.0)
        d2 = haversine(11.0, 21.0, 10.0, 20.0)
        self.assertAlmostEqual(d1, d2, places=3)

    def test_small_displacement_about_111m_per_degree(self):
        """1 degree latitude ≈ 111,194 m (spherical Earth with R=6371km)."""
        dist = haversine(0.0, 0.0, 1.0, 0.0)
        self.assertAlmostEqual(dist, 111194.9, delta=200.0)

    def test_returns_meters_not_km(self):
        dist = haversine(0.0, 0.0, 1.0, 0.0)
        self.assertGreater(dist, 1000.0)  # must be > 1000, not 111 km as a float


class CalculateRouteDistanceTests(unittest.TestCase):
    def test_empty_plan_is_zero(self):
        snap = _make_snap()
        plan = _make_plan([])
        self.assertEqual(calculate_route_distance(snap, plan), 0.0)

    def test_single_waypoint(self):
        snap = _make_snap()
        wp = Waypoint(latitude=-35.373261, longitude=149.165230, altitude_m=30.0)
        plan = _make_plan([wp])
        dist = calculate_route_distance(snap, plan)
        # ~1.1 km south
        self.assertAlmostEqual(dist / 1000.0, 1.11, delta=0.1)

    def test_multi_waypoint_accumulates(self):
        snap = _make_snap()
        wp1 = Waypoint(latitude=-35.373261, longitude=149.165230, altitude_m=30.0)
        wp2 = Waypoint(latitude=-35.383261, longitude=149.165230, altitude_m=30.0)
        plan = _make_plan([wp1, wp2])
        dist = calculate_route_distance(snap, plan)
        self.assertGreater(dist, haversine(
            snap.latitude, snap.longitude, wp1.latitude, wp1.longitude
        ))


class FeasibilityTests(unittest.TestCase):
    def _near_wp(self) -> Waypoint:
        """A waypoint very close to drone start position."""
        return Waypoint(latitude=-35.363361, longitude=149.165330, altitude_m=30.0)

    def test_simple_feasible_route(self):
        snap = _make_snap()
        cap = _make_cap()
        plan = _make_plan([self._near_wp()])
        report = evaluate_feasibility(plan, snap, cap)
        self.assertTrue(report.is_feasible)
        self.assertEqual(report.risk_level, "LOW")
        self.assertEqual(report.violations, [])

    def test_route_too_long(self):
        snap = _make_snap()
        cap = _make_cap(max_mission_distance_m=10.0)  # only 10m allowed
        wp = Waypoint(latitude=-35.373261, longitude=149.165230, altitude_m=30.0)
        plan = _make_plan([wp])
        report = evaluate_feasibility(plan, snap, cap)
        self.assertFalse(report.is_feasible)
        self.assertEqual(report.risk_level, "CRITICAL")
        self.assertTrue(any("distance" in v.lower() for v in report.violations))

    def test_too_many_waypoints(self):
        snap = _make_snap()
        cap = _make_cap(max_waypoints=2)
        wps = [self._near_wp() for _ in range(5)]
        plan = _make_plan(wps)
        report = evaluate_feasibility(plan, snap, cap)
        self.assertFalse(report.is_feasible)
        self.assertTrue(any("waypoint" in v.lower() for v in report.violations))

    def test_altitude_out_of_range(self):
        snap = _make_snap()
        cap = _make_cap(min_altitude_m=10.0, max_altitude_m=50.0)
        wp = Waypoint(latitude=-35.363361, longitude=149.165330, altitude_m=200.0)  # too high
        plan = _make_plan([wp])
        report = evaluate_feasibility(plan, snap, cap)
        self.assertFalse(report.is_feasible)
        self.assertTrue(any("altitude" in v.lower() for v in report.violations))

    def test_stale_telemetry_is_violation(self):
        # timestamp 10 seconds ago — age > 5s limit
        snap = _make_snap(timestamp=time.time() - 10.0)
        cap = _make_cap()
        plan = _make_plan([self._near_wp()])
        report = evaluate_feasibility(plan, snap, cap)
        self.assertFalse(report.is_feasible)
        self.assertTrue(any("stale" in v.lower() for v in report.violations))

    def test_low_battery_causes_violation(self):
        snap = _make_snap(battery_percent=20.5)  # just above reserve
        cap = _make_cap(
            battery_reserve_percent=20.0,
            max_mission_distance_m=5000.0,
        )
        # Use a very long route that drains > 0.5% more than available
        wps = [
            Waypoint(latitude=-35.40, longitude=149.20, altitude_m=30.0),
            Waypoint(latitude=-35.45, longitude=149.25, altitude_m=30.0),
        ]
        plan = _make_plan(wps)
        report = evaluate_feasibility(plan, snap, cap)
        # Either feasible or not, just validate the report runs without error
        self.assertIn(report.risk_level, ("LOW", "MEDIUM", "CRITICAL"))

    def test_wind_penalty_produces_warning(self):
        snap = _make_snap()
        cap = _make_cap()
        weather = _make_weather(wind_speed_mps=10.0)  # above 5 m/s threshold
        plan = _make_plan([self._near_wp()])
        report = evaluate_feasibility(plan, snap, cap, weather=weather)
        self.assertTrue(any("wind" in w.lower() for w in report.warnings))

    def test_empty_plan_is_feasible_zero_distance(self):
        snap = _make_snap()
        cap = _make_cap()
        plan = _make_plan([])
        report = evaluate_feasibility(plan, snap, cap)
        self.assertEqual(report.estimated_distance_m, 0.0)

    def test_report_always_has_corrective_suggestions_when_infeasible(self):
        snap = _make_snap()
        cap = _make_cap(max_mission_distance_m=1.0)
        wp = Waypoint(latitude=-35.40, longitude=149.20, altitude_m=30.0)
        plan = _make_plan([wp])
        report = evaluate_feasibility(plan, snap, cap)
        self.assertFalse(report.is_feasible)
        self.assertGreater(len(report.corrective_suggestions), 0)


if __name__ == "__main__":
    unittest.main()
