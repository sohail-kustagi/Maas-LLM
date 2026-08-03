import os
import json
import random

OUTPUT_PATH = os.path.join(os.getcwd(), "datasets", "phi3_finetuning.jsonl")

SYSTEM_PROMPT = (
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
    "- **CRITICAL:** You must output ONLY valid, parsable JSON. Do not include introductory text, conversational filler, or markdown formatting (do not use ```json).\n\n"
    "**Expected JSON Output Schema:**\n"
    "{\n"
    '  "frame_id": "<ID from input>",\n'
    '  "zone_assessment": {\n'
    '    "severity_level": "<Critical|High|Medium|Low>",\n'
    '    "stranded_persons_detected": <true|false>,\n'
    '    "active_rescue_in_progress": <true|false>\n'
    '  },\n'
    '  "tactical_summary": "<A strict 1-2 sentence description of detected objects on scene. E.g., 3 persons detected on infrastructure in vicinity of watercraft>",\n'
    '  "commander_recommendation": "<Hold_Position|Alert_Rescue_Teams|Continue_Recon_Pattern>"\n'
    "}"
)

def generate_scenarios(num_samples=2000):
    scenarios = []

    for i in range(1, num_samples + 1):
        frame_id = f"frame_{i:06d}"
        battery = random.randint(15, 100)
        alt = round(random.uniform(15.0, 50.0), 1)
        lat = round(random.uniform(34.0, 35.0), 5)
        lon = round(random.uniform(-118.5, -117.5), 5)
        
        # Determine scenario category
        # 0: Stranded survivor on infrastructure (Critical) -> Alert_Rescue_Teams
        # 1: Active rescue (person + watercraft) (High/Medium) -> Hold_Position
        # 2: Submerged infrastructure or vehicle only (Medium) -> Continue_Recon_Pattern
        # 3: Clear zone / routine patrol (Low) -> Continue_Recon_Pattern
        
        scenario_type = random.choices([0, 1, 2, 3], weights=[0.35, 0.25, 0.25, 0.15])[0]
        
        if scenario_type == 0:
            num_persons = random.randint(1, 5)
            num_infra = random.randint(1, 3)
            conf_p = round(random.uniform(0.70, 0.95), 2)
            conf_i = round(random.uniform(0.75, 0.98), 2)
            
            vision_payload = {
                "frame_id": frame_id,
                "timestamp": round(i * 0.2, 2),
                "detections": [
                    {"class": "person", "count": num_persons, "max_conf": conf_p, "overlap_with": ["infrastructure"]},
                    {"class": "infrastructure", "count": num_infra, "max_conf": conf_i, "condition": "submerged/flooded"}
                ]
            }
            
            assistant_response = {
                "frame_id": frame_id,
                "zone_assessment": {
                    "severity_level": "Critical",
                    "stranded_persons_detected": True,
                    "active_rescue_in_progress": False
                },
                "tactical_summary": f"{num_persons} person(s) detected stranded on infrastructure with no rescue watercraft in vicinity. Immediate extraction required.",
                "commander_recommendation": "Alert_Rescue_Teams"
            }

        elif scenario_type == 1:
            num_persons = random.randint(1, 4)
            num_boats = random.randint(1, 2)
            conf_p = round(random.uniform(0.75, 0.96), 2)
            conf_w = round(random.uniform(0.80, 0.99), 2)
            
            vision_payload = {
                "frame_id": frame_id,
                "timestamp": round(i * 0.2, 2),
                "detections": [
                    {"class": "person", "count": num_persons, "max_conf": conf_p, "proximity_to": ["watercraft"]},
                    {"class": "watercraft", "count": num_boats, "max_conf": conf_w, "status": "active_vessel"}
                ]
            }
            
            assistant_response = {
                "frame_id": frame_id,
                "zone_assessment": {
                    "severity_level": "High",
                    "stranded_persons_detected": False,
                    "active_rescue_in_progress": True
                },
                "tactical_summary": f"Active rescue operation confirmed with {num_persons} person(s) in immediate proximity to {num_boats} watercraft vessel(s).",
                "commander_recommendation": "Hold_Position"
            }

        elif scenario_type == 2:
            num_infra = random.randint(1, 4)
            num_vehicles = random.randint(0, 3)
            conf_i = round(random.uniform(0.65, 0.92), 2)
            
            detections = [{"class": "infrastructure", "count": num_infra, "max_conf": conf_i, "condition": "partially_flooded"}]
            if num_vehicles > 0:
                conf_v = round(random.uniform(0.60, 0.89), 2)
                detections.append({"class": "vehicle", "count": num_vehicles, "max_conf": conf_v, "condition": "stranded_unoccupied"})
                
            vision_payload = {
                "frame_id": frame_id,
                "timestamp": round(i * 0.2, 2),
                "detections": detections
            }
            
            summary_str = f"Detected {num_infra} structure(s) and {num_vehicles} vehicle(s) in flood zone without detected human presence." if num_vehicles > 0 else f"Detected {num_infra} structure(s) with partial flood exposure and zero visible persons."
            
            assistant_response = {
                "frame_id": frame_id,
                "zone_assessment": {
                    "severity_level": "Medium",
                    "stranded_persons_detected": False,
                    "active_rescue_in_progress": False
                },
                "tactical_summary": summary_str,
                "commander_recommendation": "Continue_Recon_Pattern"
            }

        else:
            vision_payload = {
                "frame_id": frame_id,
                "timestamp": round(i * 0.2, 2),
                "detections": []
            }
            
            assistant_response = {
                "frame_id": frame_id,
                "zone_assessment": {
                    "severity_level": "Low",
                    "stranded_persons_detected": False,
                    "active_rescue_in_progress": False
                },
                "tactical_summary": "No target anomalies, watercraft, or stranded persons detected in current reconnaissance sector.",
                "commander_recommendation": "Continue_Recon_Pattern"
            }

        user_prompt = (
            f"Telemetry Data: Altitude {alt}m, Battery {battery}%, Location ({lat}, {lon}).\n"
            f"Vision Payload: {json.dumps(vision_payload)}\n"
            "Provide tactical disaster assessment and recommendations in required JSON format."
        )
        
        scenarios.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": json.dumps(assistant_response)}
            ]
        })

    return scenarios

def save_jsonl(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
            
if __name__ == "__main__":
    print("[Dataset] Generating 2,000 synthetic Disaster Analyst training scenarios...")
    dataset = generate_scenarios(2000)
    save_jsonl(dataset, OUTPUT_PATH)
    print(f"[Dataset] Success! Saved {len(dataset)} instruction scenarios to {OUTPUT_PATH}")
