from typing import Optional

try:
    from core.types import TelemetrySnapshot, VisionEvent
    from core.weather_types import WeatherSnapshot
    from core.route_types import FeasibilityReport
except ImportError:
    from src.core.types import TelemetrySnapshot, VisionEvent
    from src.core.weather_types import WeatherSnapshot
    from src.core.route_types import FeasibilityReport
    
try:
    from core.mission_profiles import MissionProfile
except ImportError:
    from src.core.mission_profiles import MissionProfile


class AnalystNode:
    def __init__(self):
        print("[Analyst] Initialized. Awaiting triggers from Watchdog.")

    def generate_context(
        self, 
        vision_event_type: str, 
        current_telemetry: dict, 
        weather: Optional[dict] = None,
        feasibility: Optional[dict] = None,
        mission_profile: Optional[MissionProfile] = None
    ) -> str:
        """
        Takes raw triggers from Node A, telemetry, weather, and feasibility data, 
        and formulates a rich context prompt for Node C (Commander).
        """
        print(f"[Analyst] Processing anomaly: {vision_event_type}")
        
        # Extract telemetry
        alt = current_telemetry.get("alt", "Unknown")
        lat = current_telemetry.get("lat", "Unknown")
        lon = current_telemetry.get("lon", "Unknown")
        heading = current_telemetry.get("heading", "Unknown")
        
        context_prompt = ""
        if mission_profile:
            context_prompt += f"MISSION CONTEXT: {mission_profile.analyst_persona}\n\n"
            
        context_prompt += (
            f"ALERT: The edge node vision system has detected a high-confidence anomaly of type '{vision_event_type}'.\n"
            f"Current Drone Telemetry:\n"
            f"- Latitude: {lat}\n"
            f"- Longitude: {lon}\n"
            f"- Altitude: {alt}m\n"
            f"- Heading: {heading} degrees\n"
        )
        
        if weather:
            context_prompt += (
                f"\nCurrent Weather:\n"
                f"- Wind: {weather.get('wind_speed', 'Unknown')} m/s\n"
                f"- Visibility: {weather.get('visibility', 'Unknown')} m\n"
                f"- Risk Level: Lightning={weather.get('lightning', 'Unknown')}\n"
            )
            
        if feasibility:
            context_prompt += (
                f"\nRoute Feasibility Check:\n"
                f"- Is Feasible: {feasibility.get('is_feasible')}\n"
                f"- Risk Level: {feasibility.get('risk_level')}\n"
                f"- Battery Remaining: OK\n"
            )

        context_prompt += (
            f"\nThe swarm is currently holding position. As the fleet commander, you must calculate an immediate "
            f"MAVLink local offset (x, y, z) to safely maneuver the drone closer to investigate the {vision_event_type}. "
            f"Keep altitude changes minimal unless necessary. Account for wind if present. Output ONLY the JSON command."
        )
        
        return context_prompt

    def generate_event_context(
        self,
        event: VisionEvent,
        telemetry: TelemetrySnapshot,
        weather: Optional[WeatherSnapshot] = None,
        feasibility: Optional[FeasibilityReport] = None,
        mission_profile: Optional[MissionProfile] = None
    ) -> str:
        weather_dict = None
        if weather:
            weather_dict = {
                "wind_speed": weather.wind_speed_mps,
                "visibility": weather.visibility_m,
                "lightning": weather.lightning_risk
            }
            
        feasibility_dict = None
        if feasibility:
            feasibility_dict = {
                "is_feasible": feasibility.is_feasible,
                "risk_level": feasibility.risk_level
            }

        return self.generate_context(
            event.anomaly_type,
            {
                "alt": telemetry.altitude_m,
                "lat": telemetry.latitude,
                "lon": telemetry.longitude,
                "heading": telemetry.heading_deg,
            },
            weather_dict,
            feasibility_dict,
            mission_profile
        )
