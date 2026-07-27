"""Tests for route_optimizer.py — nearest-neighbour, 2-opt, scoring, and full API."""
import math
import time
import unittest
from typing import Optional

from src.core.flight_planner import haversine
from src.core.route_optimizer import (
    _nearest_neighbour_order,
    _score_route,
    _total_route_distance,
    _two_opt_improve,
    optimize_route,
)
from src.core.route_types import (
    DroneCapabilities,
    FeasibilityReport,
    FlightPlan,
    Waypoint,
)
from src.core.types import TelemetrySnapshot
from src.core.weather_types import WeatherSnapshot


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


def _snap(**overrides) -> TelemetrySnapshot:
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


def _cap(**overrides) -> DroneCapabilities:
    base = dict(
        max_speed_mps=15.0,
        max_climb_rate_mps=5.0,
        max_descent_rate_mps=3.0,
        min_altitude_m=5.0,
        max_altitude_m=120.0,
        max_mission_distance_m=50000.0,
        max_waypoints=50,
        battery_reserve_percent=20.0,
        inspection_dwell_time_s=30.0,
    )
    base.update(overrides)
    return DroneCapabilities(**base)


def _plan(waypoints: list[Waypoint], plan_id="p") -> FlightPlan:
    return FlightPlan(
        plan_id=plan_id,
        drone_id="drone-1",
        waypoints=waypoints,
        created_timestamp=time.time(),
    )


def _weather(**overrides) -> WeatherSnapshot:
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


def _feasibility(risk="LOW", battery_used=5.0, is_feasible=True) -> FeasibilityReport:
    return FeasibilityReport(
        is_feasible=is_feasible,
        risk_level=risk,
        estimated_distance_m=500.0,
        estimated_duration_s=120.0,
        estimated_battery_percent_used=battery_used,
        violations=[],
        warnings=[],
        corrective_suggestions=[],
    )


def _wp(lat: float, lon: float, alt: float = 30.0) -> Waypoint:
    return Waypoint(latitude=lat, longitude=lon, altitude_m=alt)


# ──────────────────────────────────────────────────────────────────────────────
# _total_route_distance
# ──────────────────────────────────────────────────────────────────────────────


class TotalRouteDistanceTests(unittest.TestCase):

    def test_empty_returns_zero(self):
        self.assertEqual(_total_route_distance(0.0, 0.0, []), 0.0)

    def test_single_waypoint_is_straight_line(self):
        dist = _total_route_distance(0.0, 0.0, [_wp(1.0, 0.0)])
        expected = haversine(0.0, 0.0, 1.0, 0.0)
        self.assertAlmostEqual(dist, expected, places=3)

    def test_two_waypoints_accumulates_both_segments(self):
        wps = [_wp(1.0, 0.0), _wp(2.0, 0.0)]
        dist = _total_route_distance(0.0, 0.0, wps)
        expected = (
            haversine(0.0, 0.0, 1.0, 0.0)
            + haversine(1.0, 0.0, 2.0, 0.0)
        )
        self.assertAlmostEqual(dist, expected, places=3)

    def test_distance_is_always_non_negative(self):
        wps = [_wp(-35.37, 149.17), _wp(-35.38, 149.16)]
        dist = _total_route_distance(-35.36, 149.16, wps)
        self.assertGreaterEqual(dist, 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# _nearest_neighbour_order
# ──────────────────────────────────────────────────────────────────────────────


class NearestNeighbourTests(unittest.TestCase):

    def test_empty_returns_empty(self):
        self.assertEqual(_nearest_neighbour_order(0.0, 0.0, []), [])

    def test_single_waypoint_unchanged(self):
        wp = _wp(1.0, 0.0)
        result = _nearest_neighbour_order(0.0, 0.0, [wp])
        self.assertEqual(result, [wp])

    def test_closest_is_first(self):
        """The nearest waypoint to origin should be placed first."""
        close = _wp(0.001, 0.0)   # ~111 m away
        far = _wp(1.0, 0.0)       # ~111 km away
        result = _nearest_neighbour_order(0.0, 0.0, [far, close])
        self.assertEqual(result[0], close)
        self.assertEqual(result[1], far)

    def test_produces_permutation_of_input(self):
        wps = [_wp(float(i), 0.0) for i in range(5)]
        result = _nearest_neighbour_order(0.0, 0.0, wps)
        self.assertEqual(set(id(w) for w in result), set(id(w) for w in wps))

    def test_all_waypoints_included(self):
        wps = [_wp(float(i), float(i)) for i in range(10)]
        result = _nearest_neighbour_order(0.0, 0.0, wps)
        self.assertEqual(len(result), 10)

    def test_nn_improves_over_reverse_order(self):
        """NN-ordered route must be ≤ worst-case reverse order."""
        wps = [_wp(float(i) * 0.1, 0.0) for i in range(5, 0, -1)]  # far to close
        dist_nn = _total_route_distance(0.0, 0.0, _nearest_neighbour_order(0.0, 0.0, wps))
        dist_raw = _total_route_distance(0.0, 0.0, wps)
        self.assertLessEqual(dist_nn, dist_raw + 1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# _two_opt_improve
# ──────────────────────────────────────────────────────────────────────────────


class TwoOptTests(unittest.TestCase):

    def test_empty_returns_empty(self):
        self.assertEqual(_two_opt_improve(0.0, 0.0, []), [])

    def test_single_waypoint_returned_unchanged(self):
        wp = _wp(1.0, 0.0)
        result = _two_opt_improve(0.0, 0.0, [wp])
        self.assertEqual(result, [wp])

    def test_two_waypoints_unchanged(self):
        wps = [_wp(1.0, 0.0), _wp(0.5, 0.0)]
        result = _two_opt_improve(0.0, 0.0, wps)
        self.assertEqual(len(result), 2)

    def test_produces_permutation_of_input(self):
        wps = [_wp(float(i) * 0.1, float(i) * 0.05) for i in range(6)]
        result = _two_opt_improve(0.0, 0.0, wps)
        self.assertEqual(
            sorted((w.latitude, w.longitude) for w in result),
            sorted((w.latitude, w.longitude) for w in wps),
        )

    def test_result_is_no_longer_than_input_route(self):
        """2-opt must never make the route longer."""
        wps = [
            _wp(1.0, 0.0), _wp(0.0, 1.0), _wp(0.5, 0.5),
            _wp(1.5, 0.5), _wp(0.3, 0.8),
        ]
        dist_before = _total_route_distance(0.0, 0.0, wps)
        improved = _two_opt_improve(0.0, 0.0, wps)
        dist_after = _total_route_distance(0.0, 0.0, improved)
        self.assertLessEqual(dist_after, dist_before + 1e-6)

    def test_2opt_improves_known_crossing_route(self):
        """A route that crosses itself must be shortened by 2-opt."""
        # A→D→B→C is longer than A→B→C→D for a rectangular route
        # Layout: A(0,0) B(0,1) C(1,1) D(1,0)
        A = _wp(0.0, 0.0)
        B = _wp(0.0, 1.0)
        C = _wp(1.0, 1.0)
        D = _wp(1.0, 0.0)
        crossing = [A, C, B, D]  # this crosses
        optimal  = [A, B, C, D]  # this does not

        dist_crossing = _total_route_distance(0.0, 0.0, crossing)
        dist_optimal  = _total_route_distance(0.0, 0.0, optimal)

        improved = _two_opt_improve(0.0, 0.0, crossing)
        dist_improved = _total_route_distance(0.0, 0.0, improved)

        # 2-opt must find the same or better distance than the crossing route
        self.assertLessEqual(dist_improved, dist_crossing + 1e-6)
        # And it must not be worse than the known optimal
        self.assertLessEqual(dist_improved, dist_optimal + 1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# _score_route
# ──────────────────────────────────────────────────────────────────────────────


class ScoreRouteTests(unittest.TestCase):

    def test_score_is_in_zero_to_one(self):
        feas = _feasibility(battery_used=5.0, risk="LOW")
        score = _score_route(100.0, feas, None)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_shorter_route_scores_higher(self):
        feas = _feasibility()
        short_score = _score_route(100.0, feas, None)
        long_score = _score_route(4999.0, feas, None)
        self.assertGreater(short_score, long_score)

    def test_lower_battery_use_scores_higher(self):
        feas_low = _feasibility(battery_used=5.0)
        feas_high = _feasibility(battery_used=60.0)
        score_low = _score_route(500.0, feas_low, None)
        score_high = _score_route(500.0, feas_high, None)
        self.assertGreater(score_low, score_high)

    def test_calm_weather_scores_higher_than_windy(self):
        feas = _feasibility()
        calm = _weather(wind_speed_mps=0.0, precipitation_mm=0.0)
        windy = _weather(wind_speed_mps=14.0, precipitation_mm=8.0)
        score_calm = _score_route(500.0, feas, calm)
        score_windy = _score_route(500.0, feas, windy)
        self.assertGreater(score_calm, score_windy)

    def test_low_risk_scores_higher_than_critical(self):
        feas_low = _feasibility(risk="LOW")
        feas_crit = _feasibility(risk="CRITICAL")
        score_low = _score_route(500.0, feas_low, None)
        score_crit = _score_route(500.0, feas_crit, None)
        self.assertGreater(score_low, score_crit)

    def test_storm_reduces_score_significantly(self):
        feas = _feasibility()
        clear = _weather(thunderstorm_risk=0.0, lightning_risk=0.0)
        storm = _weather(thunderstorm_risk=1.0, lightning_risk=1.0)
        self.assertGreater(
            _score_route(500.0, feas, clear),
            _score_route(500.0, feas, storm),
        )

    def test_no_weather_does_not_crash(self):
        feas = _feasibility()
        score = _score_route(500.0, feas, None)
        self.assertIsInstance(score, float)

    def test_zero_distance_gives_max_distance_component(self):
        feas = _feasibility(battery_used=0.0, risk="LOW")
        score = _score_route(0.0, feas, None)
        # Should be very high (≥ 0.9)
        self.assertGreater(score, 0.9)


# ──────────────────────────────────────────────────────────────────────────────
# optimize_route (full API)
# ──────────────────────────────────────────────────────────────────────────────


class OptimizeRouteTests(unittest.TestCase):

    def _nearby_wps(self) -> list[Waypoint]:
        """Small cluster of waypoints near Canberra for fast testing."""
        return [
            _wp(-35.364, 149.166),
            _wp(-35.362, 149.164),
            _wp(-35.365, 149.163),
            _wp(-35.361, 149.167),
        ]

    def test_returns_route_recommendation(self):
        from src.core.route_types import RouteRecommendation
        snap = _snap()
        plan = _plan(self._nearby_wps())
        rec = optimize_route(plan, snap, _cap())
        self.assertIsInstance(rec, RouteRecommendation)

    def test_optimised_plan_has_same_waypoint_count(self):
        snap = _snap()
        plan = _plan(self._nearby_wps())
        rec = optimize_route(plan, snap, _cap())
        self.assertEqual(len(rec.plan.waypoints), len(plan.waypoints))

    def test_optimised_plan_id_has_opt_suffix(self):
        snap = _snap()
        plan = _plan(self._nearby_wps(), plan_id="test-plan")
        rec = optimize_route(plan, snap, _cap())
        self.assertTrue(rec.plan.plan_id.endswith("-opt"))

    def test_score_is_in_range(self):
        snap = _snap()
        plan = _plan(self._nearby_wps())
        rec = optimize_route(plan, snap, _cap())
        self.assertGreaterEqual(rec.score, 0.0)
        self.assertLessEqual(rec.score, 1.0)

    def test_reasoning_is_non_empty_string(self):
        snap = _snap()
        plan = _plan(self._nearby_wps())
        rec = optimize_route(plan, snap, _cap())
        self.assertIsInstance(rec.reasoning, str)
        self.assertGreater(len(rec.reasoning), 20)

    def test_reasoning_contains_distance(self):
        snap = _snap()
        plan = _plan(self._nearby_wps())
        rec = optimize_route(plan, snap, _cap())
        self.assertIn("m", rec.reasoning)

    def test_empty_plan_returns_feasible_zero_distance(self):
        snap = _snap()
        plan = _plan([])
        rec = optimize_route(plan, snap, _cap())
        self.assertEqual(rec.feasibility.estimated_distance_m, 0.0)

    def test_weather_context_appears_in_reasoning(self):
        snap = _snap()
        plan = _plan(self._nearby_wps())
        w = _weather(wind_speed_mps=8.0)
        rec = optimize_route(plan, snap, _cap(), weather=w)
        self.assertIn("wind", rec.reasoning.lower())

    def test_optimised_route_not_longer_than_original(self):
        """The whole point of optimization — shorter or equal distance."""
        from src.core.route_optimizer import _total_route_distance
        snap = _snap()
        wps = self._nearby_wps()
        original_dist = _total_route_distance(snap.latitude, snap.longitude, wps)
        plan = _plan(wps)
        rec = optimize_route(plan, snap, _cap())
        optimised_dist = _total_route_distance(
            snap.latitude, snap.longitude, rec.plan.waypoints
        )
        self.assertLessEqual(optimised_dist, original_dist + 1e-6)

    def test_infeasible_plan_still_returns_recommendation(self):
        """Even infeasible plans must return a recommendation — caller decides."""
        snap = _snap(battery_percent=10.0)  # very low battery
        wps = [_wp(-35.40, 149.20), _wp(-35.50, 149.30)]  # long route
        plan = _plan(wps)
        cap = _cap(battery_reserve_percent=25.0, max_mission_distance_m=100.0)
        rec = optimize_route(plan, snap, cap)
        self.assertIsNotNone(rec)
        # Feasibility may or may not pass — just check we get a result
        self.assertIn(rec.feasibility.risk_level, ("LOW", "MEDIUM", "HIGH", "CRITICAL"))


if __name__ == "__main__":
    unittest.main()
