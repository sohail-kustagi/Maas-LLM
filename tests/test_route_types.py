"""Tests for Waypoint, FlightPlan, DroneCapabilities, FeasibilityReport, RouteRecommendation."""
import unittest

from src.core.route_types import (
    DroneCapabilities,
    FeasibilityReport,
    FlightPlan,
    RouteConstraints,
    RouteRecommendation,
    Waypoint,
)


def _make_waypoint(**overrides) -> Waypoint:
    base = dict(latitude=-35.363, longitude=149.165, altitude_m=30.0)
    base.update(overrides)
    return Waypoint(**base)


def _make_capabilities(**overrides) -> DroneCapabilities:
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


class WaypointTests(unittest.TestCase):
    def test_defaults(self):
        wp = _make_waypoint()
        self.assertEqual(wp.hold_time_s, 0.0)
        self.assertEqual(wp.accept_radius_m, 5.0)

    def test_custom_values(self):
        wp = _make_waypoint(hold_time_s=10.0, accept_radius_m=2.0)
        self.assertEqual(wp.hold_time_s, 10.0)
        self.assertEqual(wp.accept_radius_m, 2.0)

    def test_is_frozen(self):
        wp = _make_waypoint()
        with self.assertRaises((AttributeError, TypeError)):
            wp.altitude_m = 999.0  # type: ignore[misc]


class FlightPlanTests(unittest.TestCase):
    def test_empty_waypoints(self):
        plan = FlightPlan(
            plan_id="plan-001",
            drone_id="drone-1",
            waypoints=[],
            created_timestamp=0.0,
        )
        self.assertEqual(len(plan.waypoints), 0)

    def test_multiple_waypoints(self):
        wps = [_make_waypoint(altitude_m=float(a)) for a in (20, 30, 40)]
        plan = FlightPlan("p", "d", wps, 0.0)
        self.assertEqual(len(plan.waypoints), 3)
        self.assertEqual(plan.waypoints[2].altitude_m, 40.0)


class DroneCapabilitiesTests(unittest.TestCase):
    def test_all_fields_stored(self):
        cap = _make_capabilities()
        self.assertEqual(cap.max_speed_mps, 15.0)
        self.assertEqual(cap.max_waypoints, 20)
        self.assertEqual(cap.battery_reserve_percent, 20.0)


class FeasibilityReportTests(unittest.TestCase):
    def test_feasible_report(self):
        report = FeasibilityReport(
            is_feasible=True,
            risk_level="LOW",
            estimated_distance_m=500.0,
            estimated_duration_s=120.0,
            estimated_battery_percent_used=5.0,
            violations=[],
            warnings=[],
            corrective_suggestions=[],
        )
        self.assertTrue(report.is_feasible)
        self.assertEqual(report.risk_level, "LOW")
        self.assertEqual(report.violations, [])

    def test_infeasible_report_has_violations(self):
        report = FeasibilityReport(
            is_feasible=False,
            risk_level="CRITICAL",
            estimated_distance_m=9999.0,
            estimated_duration_s=9999.0,
            estimated_battery_percent_used=200.0,
            violations=["Battery reserve exceeded", "Distance too far"],
            warnings=["High wind"],
            corrective_suggestions=["Reduce route"],
        )
        self.assertFalse(report.is_feasible)
        self.assertEqual(len(report.violations), 2)
        self.assertEqual(report.risk_level, "CRITICAL")


class RouteConstraintsTests(unittest.TestCase):
    def test_defaults(self):
        constraints = RouteConstraints()
        self.assertIsNone(constraints.geofence_polygon)
        self.assertEqual(constraints.min_separation_m, 10.0)
        self.assertIsNone(constraints.blocked_cells)


class RouteRecommendationTests(unittest.TestCase):
    def test_recommendation_fields(self):
        wp = _make_waypoint()
        plan = FlightPlan("p", "d", [wp], 0.0)
        feasibility = FeasibilityReport(
            is_feasible=True, risk_level="LOW",
            estimated_distance_m=100.0, estimated_duration_s=60.0,
            estimated_battery_percent_used=2.0, violations=[], warnings=[],
            corrective_suggestions=[],
        )
        rec = RouteRecommendation(plan=plan, feasibility=feasibility, score=0.92, reasoning="Best route.")
        self.assertAlmostEqual(rec.score, 0.92)
        self.assertEqual(rec.reasoning, "Best route.")


if __name__ == "__main__":
    unittest.main()
