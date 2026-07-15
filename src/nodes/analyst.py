class AnalystNode:
    def __init__(self):
        print("[Analyst] Initialized. Awaiting triggers from Watchdog.")

    def generate_context(self, vision_event_type: str, current_telemetry: dict):
        """
        Takes raw triggers from Node A and raw telemetry data, 
        and formulates a rich context prompt for Node C (Commander).
        """
        print(f"[Analyst] Processing anomaly: {vision_event_type}")
        
        # Extract telemetry
        alt = current_telemetry.get("alt", "Unknown")
        lat = current_telemetry.get("lat", "Unknown")
        lon = current_telemetry.get("lon", "Unknown")
        heading = current_telemetry.get("heading", "Unknown")
        
        context_prompt = (
            f"ALERT: The edge node vision system has detected a high-confidence anomaly of type '{vision_event_type}'.\n"
            f"Current Drone Telemetry:\n"
            f"- Latitude: {lat}\n"
            f"- Longitude: {lon}\n"
            f"- Altitude: {alt}m\n"
            f"- Heading: {heading} degrees\n\n"
            f"The swarm is currently holding position. As the fleet commander, you must calculate an immediate "
            f"MAVLink local offset (x, y, z) to safely maneuver the drone closer to investigate the {vision_event_type}. "
            f"Keep altitude changes minimal unless necessary. Output ONLY the JSON command."
        )
        
        return context_prompt
