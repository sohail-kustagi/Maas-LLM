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
        self.system_prompt = (
            "**Role:** You are the Lead Disaster Analyst AI for an autonomous Search and Rescue (SAR) drone system. "
            "You operate as the critical reasoning layer between the drone's raw computer vision pipeline and the flight Commander node.\n\n"
            "**Input Data:** You will receive telemetry data and structured JSON payloads from an aerial YOLO-SAHI vision model. "
            "The vision model detects four specific classes from a top-down perspective:\n"
            "1. `infrastructure` (submerged or intact buildings/houses/docks)\n"
            "2. `person` (survivors or rescue personnel)\n"
            "3. `vehicle` (ground transport like cars/trucks)\n"
            "4. `watercraft` (boats or rescue vessels)\n\n"
            "**Objectives:**\n"
            "1. **Analyze Spatial Context:** Interpret the relationships between detected objects. For example, a `person` bounding box overlapping an `infrastructure` bounding box indicates a stranded survivor on a roof. A `person` next to a `watercraft` indicates an active rescue.\n"
            "2. **Triage & Prioritize:** Assign a threat severity level (Critical, High, Medium, Low) to the current drone visual frame based on the presence and context of stranded persons.\n"
            "3. **Report:** Generate a structured, actionable intelligence brief for the Commander node to determine the next flight path or alert protocol.\n\n"
            "**Constraints:**\n"
            "- You must rely ONLY on the provided vision payload. Do not hallucinate objects or events that are not explicitly detected.\n"
            "- You are functioning in a high-speed pipeline. Keep your reasoning concise.\n"
            "- **CRITICAL:** You must output ONLY valid, parsable JSON. Do not include introductory text, conversational filler, or markdown formatting.\n"
        )
        print("[Analyst] Initialized Lead Disaster Analyst AI with 4-class SAR schema.")

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
        and formulates a rich structured context prompt for Node C (Commander).
        """
        print(f"[Analyst] Processing anomaly: {vision_event_type}")
        
        if isinstance(current_telemetry, dict):
            alt = current_telemetry.get("alt", "Unknown")
            lat = current_telemetry.get("lat", "Unknown")
            lon = current_telemetry.get("lon", "Unknown")
            heading = current_telemetry.get("heading", "Unknown")
        else:
            alt = getattr(current_telemetry, "altitude_m", "Unknown")
            lat = getattr(current_telemetry, "latitude", "Unknown")
            lon = getattr(current_telemetry, "longitude", "Unknown")
            heading = getattr(current_telemetry, "heading_deg", "Unknown")
        
        context_prompt = f"{self.system_prompt}\n\n"
        if mission_profile:
            context_prompt += f"MISSION CONTEXT: {mission_profile.analyst_persona}\n\n"
            
        context_prompt += (
            f"ALERT: The edge vision system has detected high-confidence anomalies involving '{vision_event_type}'.\n"
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
            )

        context_prompt += (
            f"\nBased strictly on the telemetry and vision detection of '{vision_event_type}', generate the MAVLink routing "
            f"command in ONLY valid JSON."
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
