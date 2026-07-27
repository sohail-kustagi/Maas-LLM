"""
route_optimizer.py — Deterministic route optimization for single-drone missions.

Implements:
  1. Nearest-neighbour waypoint ordering (greedy)
  2. 2-opt improvement (iterative swap until no gain)
  3. Route scoring (distance, battery, weather penalty, risk)
  4. Public API: optimize_route(plan, snapshot, capabilities, weather)

The LLM is NEVER involved here. This is pure deterministic geometry.
"""
from __future__ import annotations

import math
from typing import Optional

from .flight_planner import evaluate_feasibility, haversine
from .route_types import (
    DroneCapabilities,
    FeasibilityReport,
    FlightPlan,
    RouteRecommendation,
    Waypoint,
)
from .types import TelemetrySnapshot
from .weather_types import WeatherSnapshot

# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _total_route_distance(
    origin_lat: float,
    origin_lon: float,
    waypoints: list[Waypoint],
) -> float:
    """Sum of segment distances (origin → wp0 → wp1 → … → wpN) in metres."""
    if not waypoints:
        return 0.0
    total = haversine(origin_lat, origin_lon, waypoints[0].latitude, waypoints[0].longitude)
    for i in range(len(waypoints) - 1):
        total += haversine(
            waypoints[i].latitude, waypoints[i].longitude,
            waypoints[i + 1].latitude, waypoints[i + 1].longitude,
        )
    return total


def _nearest_neighbour_order(
    origin_lat: float,
    origin_lon: float,
    waypoints: list[Waypoint],
) -> list[Waypoint]:
    """Greedy nearest-neighbour ordering starting from origin.

    Time complexity: O(n²) — acceptable for n ≤ 100 waypoints.
    """
    if not waypoints:
        return []

    remaining = list(waypoints)
    ordered: list[Waypoint] = []
    current_lat, current_lon = origin_lat, origin_lon

    while remaining:
        # Find the closest unvisited waypoint
        nearest_idx = min(
            range(len(remaining)),
            key=lambda i: haversine(
                current_lat, current_lon,
                remaining[i].latitude, remaining[i].longitude,
            ),
        )
        wp = remaining.pop(nearest_idx)
        ordered.append(wp)
        current_lat, current_lon = wp.latitude, wp.longitude

    return ordered


def _two_opt_improve(
    origin_lat: float,
    origin_lon: float,
    waypoints: list[Waypoint],
    max_iterations: int = 200,
) -> list[Waypoint]:
    """2-opt local search improvement.

    Iteratively reverses sub-sequences to find shorter total route length.
    Stops when no improvement is found in a full pass, or after max_iterations.

    Time complexity per iteration: O(n²).
    """
    if len(waypoints) < 3:
        return list(waypoints)

    best = list(waypoints)
    best_dist = _total_route_distance(origin_lat, origin_lon, best)
    improved = True
    iteration = 0

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        n = len(best)
        for i in range(n - 1):
            for j in range(i + 2, n):
                # Reverse the sub-segment between i+1 and j
                candidate = best[:i + 1] + best[i + 1:j + 1][::-1] + best[j + 1:]
                dist = _total_route_distance(origin_lat, origin_lon, candidate)
                if dist < best_dist - 1e-6:  # 1mm numerical tolerance
                    best = candidate
                    best_dist = dist
                    improved = True
                    break  # restart inner loop after improvement
            if improved:
                break

    return best


def _score_route(
    distance_m: float,
    feasibility: FeasibilityReport,
    weather: Optional[WeatherSnapshot],
) -> float:
    """Compute a [0, 1] composite score. Higher is better.

    Weights:
      - Distance (shorter is better): 40%
      - Battery usage (lower is better): 30%
      - Weather penalty (calmer is better): 20%
      - Risk level (lower is better): 10%
    """
    # Normalise distance: assume anything over 5 km scores 0
    MAX_DISTANCE_M = 5000.0
    distance_score = max(0.0, 1.0 - distance_m / MAX_DISTANCE_M)

    # Battery score: invert usage percentage
    battery_used = feasibility.estimated_battery_percent_used
    battery_score = max(0.0, 1.0 - battery_used / 100.0)

    # Weather penalty
    weather_score = 1.0
    if weather is not None:
        # Wind penalty: 0 → 1 as wind approaches 15 m/s
        wind_penalty = min(1.0, weather.wind_speed_mps / 15.0)
        # Precipitation penalty
        precip_penalty = min(1.0, weather.precipitation_mm / 10.0)
        # Thunderstorm / lightning hard penalty
        storm_penalty = max(weather.thunderstorm_risk, weather.lightning_risk)
        weather_score = max(0.0, 1.0 - 0.4 * wind_penalty - 0.3 * precip_penalty - 0.3 * storm_penalty)

    # Risk score
    risk_map = {"LOW": 1.0, "MEDIUM": 0.6, "HIGH": 0.3, "CRITICAL": 0.0}
    risk_score = risk_map.get(feasibility.risk_level, 0.5)

    return (
        0.40 * distance_score
        + 0.30 * battery_score
        + 0.20 * weather_score
        + 0.10 * risk_score
    )


def _make_reasoning(
    original_distance_m: float,
    optimised_distance_m: float,
    score: float,
    feasibility: FeasibilityReport,
    weather: Optional[WeatherSnapshot],
) -> str:
    """Build a human-readable reasoning string for the recommendation."""
    saving_m = original_distance_m - optimised_distance_m
    saving_pct = (saving_m / original_distance_m * 100) if original_distance_m > 0 else 0.0

    parts = [
        f"Nearest-neighbour + 2-opt optimisation reduced route by "
        f"{saving_m:.0f} m ({saving_pct:.1f}%). "
        f"Estimated distance: {optimised_distance_m:.0f} m, "
        f"battery: {feasibility.estimated_battery_percent_used:.1f}%, "
        f"risk: {feasibility.risk_level}.",
    ]

    if weather is not None:
        parts.append(
            f"Weather at route: wind {weather.wind_speed_mps:.1f} m/s, "
            f"visibility {weather.visibility_m:.0f} m, "
            f"precipitation {weather.precipitation_mm:.1f} mm."
        )

    if feasibility.warnings:
        parts.append("Warnings: " + "; ".join(feasibility.warnings) + ".")

    parts.append(f"Composite route score: {score:.2f}/1.00.")
    return " ".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def optimize_route(
    plan: FlightPlan,
    snapshot: TelemetrySnapshot,
    capabilities: DroneCapabilities,
    weather: Optional[WeatherSnapshot] = None,
) -> RouteRecommendation:
    """Optimise a flight plan and return a scored RouteRecommendation.

    Steps:
      1. Nearest-neighbour ordering from the drone's current position
      2. 2-opt local improvement
      3. Feasibility evaluation on the optimised route
      4. Score calculation and reasoning generation

    Returns a RouteRecommendation regardless of feasibility — the caller
    must check `recommendation.feasibility.is_feasible` before executing.
    """
    origin_lat = snapshot.latitude
    origin_lon = snapshot.longitude

    # Step 1: record original distance for reporting
    original_distance_m = _total_route_distance(origin_lat, origin_lon, plan.waypoints)

    # Step 2: nearest-neighbour ordering
    ordered_wps = _nearest_neighbour_order(origin_lat, origin_lon, plan.waypoints)

    # Step 3: 2-opt improvement
    optimised_wps = _two_opt_improve(origin_lat, origin_lon, ordered_wps)
    optimised_distance_m = _total_route_distance(origin_lat, origin_lon, optimised_wps)

    # Step 4: build optimised plan
    optimised_plan = FlightPlan(
        plan_id=f"{plan.plan_id}-opt",
        drone_id=plan.drone_id,
        waypoints=optimised_wps,
        created_timestamp=plan.created_timestamp,
    )

    # Step 5: evaluate feasibility on the optimised plan
    feasibility = evaluate_feasibility(optimised_plan, snapshot, capabilities, weather=weather)

    # Step 6: score
    score = _score_route(optimised_distance_m, feasibility, weather)

    # Step 7: reasoning
    reasoning = _make_reasoning(
        original_distance_m, optimised_distance_m, score, feasibility, weather
    )

    return RouteRecommendation(
        plan=optimised_plan,
        feasibility=feasibility,
        score=score,
        reasoning=reasoning,
    )
