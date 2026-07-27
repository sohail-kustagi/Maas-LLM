from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass(frozen=True)
class Waypoint:
    latitude: float
    longitude: float
    altitude_m: float
    hold_time_s: float = 0.0
    accept_radius_m: float = 5.0

@dataclass(frozen=True)
class FlightPlan:
    plan_id: str
    drone_id: str
    waypoints: List[Waypoint]
    created_timestamp: float

@dataclass(frozen=True)
class DroneCapabilities:
    max_speed_mps: float
    max_climb_rate_mps: float
    max_descent_rate_mps: float
    min_altitude_m: float
    max_altitude_m: float
    max_mission_distance_m: float
    max_waypoints: int
    battery_reserve_percent: float
    inspection_dwell_time_s: float

@dataclass(frozen=True)
class RouteConstraints:
    geofence_polygon: Optional[List[tuple[float, float]]] = None
    min_separation_m: float = 10.0
    blocked_cells: Optional[List[tuple[float, float]]] = None

@dataclass(frozen=True)
class FeasibilityReport:
    is_feasible: bool
    risk_level: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    estimated_distance_m: float
    estimated_duration_s: float
    estimated_battery_percent_used: float
    violations: List[str]
    warnings: List[str]
    corrective_suggestions: List[str]

@dataclass(frozen=True)
class RouteRecommendation:
    plan: FlightPlan
    feasibility: FeasibilityReport
    score: float
    reasoning: str
