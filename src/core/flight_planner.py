import math
import time
from typing import List, Optional
from .types import TelemetrySnapshot
from .route_types import FlightPlan, DroneCapabilities, FeasibilityReport, Waypoint
from .weather_types import WeatherSnapshot

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance in meters between two points on the earth."""
    R = 6371000.0  # Earth radius in meters
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def calculate_route_distance(telemetry: TelemetrySnapshot, plan: FlightPlan) -> float:
    if not plan.waypoints:
        return 0.0
    
    total_distance = 0.0
    curr_lat, curr_lon = telemetry.latitude, telemetry.longitude
    
    for wp in plan.waypoints:
        total_distance += haversine(curr_lat, curr_lon, wp.latitude, wp.longitude)
        curr_lat, curr_lon = wp.latitude, wp.longitude
        
    return total_distance

def evaluate_feasibility(
    plan: FlightPlan, 
    telemetry: TelemetrySnapshot, 
    capabilities: DroneCapabilities,
    weather: Optional[WeatherSnapshot] = None
) -> FeasibilityReport:
    violations = []
    warnings = []
    corrective_suggestions = []
    
    # 1. Telemetry Freshness
    age = telemetry.age_seconds()
    if age > 5.0:
        violations.append(f"Stale telemetry (age {age:.1f}s > 5s limit)")
        
    # 2. Distance and Duration
    dist = calculate_route_distance(telemetry, plan)
    if dist > capabilities.max_mission_distance_m:
        violations.append(f"Route distance {dist:.1f}m exceeds max mission distance {capabilities.max_mission_distance_m}m")
        
    # Assume cruising speed is 80% of max speed
    cruise_speed = capabilities.max_speed_mps * 0.8
    duration_s = (dist / cruise_speed) if cruise_speed > 0 else 0
    
    # Add dwell time
    total_dwell = sum(wp.hold_time_s for wp in plan.waypoints)
    duration_s += total_dwell
    
    # Add inspection dwell time
    duration_s += capabilities.inspection_dwell_time_s
    
    # 3. Battery Estimation
    # Very simple conservative model: 1% battery per 1000m + 1% per 60s hover
    # Wind penalty: add 20% to battery usage if wind > 5 m/s
    wind_penalty = 1.0
    if weather and weather.wind_speed_mps > 5.0:
        wind_penalty = 1.2
        warnings.append("High wind detected, applying 20% battery penalty.")
        
    battery_used = ((dist / 1000.0) * 1.0 + (total_dwell / 60.0) * 1.0) * wind_penalty
    
    current_battery = telemetry.battery_percent if telemetry.battery_percent is not None else 100.0
    remaining_battery = current_battery - battery_used
    
    if remaining_battery < capabilities.battery_reserve_percent:
        violations.append(f"Estimated remaining battery {remaining_battery:.1f}% is below reserve {capabilities.battery_reserve_percent}%")
        
    # 4. Waypoint Bounds
    if len(plan.waypoints) > capabilities.max_waypoints:
        violations.append(f"Too many waypoints ({len(plan.waypoints)} > {capabilities.max_waypoints})")
        
    for i, wp in enumerate(plan.waypoints):
        if not (capabilities.min_altitude_m <= wp.altitude_m <= capabilities.max_altitude_m):
            violations.append(f"Waypoint {i} altitude {wp.altitude_m}m outside allowed range [{capabilities.min_altitude_m}, {capabilities.max_altitude_m}]m")
            
    is_feasible = len(violations) == 0
    risk_level = "LOW"
    if len(violations) > 0:
        risk_level = "CRITICAL"
    elif len(warnings) > 0:
        risk_level = "MEDIUM"
        
    if not is_feasible:
        corrective_suggestions.append("Consider reducing route distance or dropping waypoints.")
        
    return FeasibilityReport(
        is_feasible=is_feasible,
        risk_level=risk_level,
        estimated_distance_m=dist,
        estimated_duration_s=duration_s,
        estimated_battery_percent_used=battery_used,
        violations=violations,
        warnings=warnings,
        corrective_suggestions=corrective_suggestions
    )
