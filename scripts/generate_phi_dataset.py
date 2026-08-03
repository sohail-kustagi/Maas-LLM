import os
import json
import random

OUTPUT_PATH = os.path.join(os.getcwd(), "datasets", "phi3_finetuning.jsonl")

def generate_scenarios(num_samples=2000):
    system_prompt = (
        "You are the tactical commander for a MAAS disaster drone.\n"
        "Output ONLY a raw JSON object with NO markdown, NO comments, NO extra text.\n"
        "Required keys (all with double-quoted strings and float values):\n"
        '  command: always "SET_POSITION_TARGET_LOCAL_NED"\n'
        '  target_system: 1\n'
        '  target_component: 1\n'
        '  x: float (meters North, max 100)\n'
        '  y: float (meters East, max 100)\n'
        '  z: float (negative = up, e.g. -20.0 for 20m altitude)\n'
        '  reasoning: short string\n'
        "Start your response with { and end with }. No backticks. No extra lines."
    )

    scenarios = []

    for _ in range(num_samples):
        # Randomize drone state
        battery = random.randint(5, 100)
        alt = round(random.uniform(5.0, 60.0), 1)
        
        # Randomize vision events with realistic coordinates
        x_target = round(random.uniform(-50.0, 50.0), 1)
        y_target = round(random.uniform(-50.0, 50.0), 1)
        
        vision_options = [
            ("No notable objects detected.", None),
            (f"Detected fire at X: {x_target}, Y: {y_target} with 0.89 confidence", "fire"),
            (f"Detected flood_water at X: {x_target}, Y: {y_target} with 0.94 confidence", "flood_water"),
            (f"Detected person at X: {x_target}, Y: {y_target} with 0.77 confidence", "person"),
            (f"Detected rescue_tag at X: {x_target}, Y: {y_target} with 0.91 confidence", "rescue_tag"),
        ]
        
        # 30% chance of nothing, 70% chance of an event
        if random.random() < 0.3:
            vision_text, event = vision_options[0]
        else:
            vision_text, event = random.choice(vision_options[1:])
        
        user_prompt = f"Telemetry Data: Altitude {alt}m, Battery {battery}%.\nYOLO Vision Report: {vision_text}.\nWhat is your tactical command?"
        
        # Tactical Logic Rules Engine (this is what we are teaching the LLM)
        if battery < 20:
            assistant = {
                "command": "SET_POSITION_TARGET_LOCAL_NED",
                "target_system": 1,
                "target_component": 1,
                "x": 0.0,
                "y": 0.0,
                "z": -10.0,
                "reasoning": "CRITICAL BATTERY: Returning to home coordinates immediately to swap battery."
            }
        elif event == "fire":
            assistant = {
                "command": "SET_POSITION_TARGET_LOCAL_NED",
                "target_system": 1,
                "target_component": 1,
                "x": x_target,
                "y": y_target,
                "z": -30.0,
                "reasoning": "Fire detected. Moving to coordinates at safe high altitude to monitor spread and alert ground crews."
            }
        elif event == "flood_water":
            assistant = {
                "command": "SET_POSITION_TARGET_LOCAL_NED",
                "target_system": 1,
                "target_component": 1,
                "x": x_target,
                "y": y_target,
                "z": -20.0,
                "reasoning": "Flood detected. Loitering over the area to measure water level and assess damage."
            }
        elif event == "person":
            assistant = {
                "command": "SET_POSITION_TARGET_LOCAL_NED",
                "target_system": 1,
                "target_component": 1,
                "x": x_target,
                "y": y_target,
                "z": -10.0,
                "reasoning": "Survivor detected! Descending rapidly to 10m to establish visual and prep rescue payload drop."
            }
        elif event == "rescue_tag":
            assistant = {
                "command": "SET_POSITION_TARGET_LOCAL_NED",
                "target_system": 1,
                "target_component": 1,
                "x": x_target,
                "y": y_target,
                "z": -15.0,
                "reasoning": "SOS/Rescue Tag detected. Locking position to alert Search and Rescue command center."
            }
        else:
            # Routine Patrol: move in a random grid direction
            x_move = round(random.uniform(5.0, 15.0), 1)
            y_move = round(random.uniform(-10.0, 10.0), 1)
            assistant = {
                "command": "SET_POSITION_TARGET_LOCAL_NED",
                "target_system": 1,
                "target_component": 1,
                "x": x_move,
                "y": y_move,
                "z": -25.0,
                "reasoning": "Routine patrol. No targets detected, executing standard search grid progression."
            }
            
        # Serialize the assistant's JSON without pretty printing so it strictly matches the format
        assistant_str = json.dumps(assistant)
        
        scenarios.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_str}
            ]
        })
        
    return scenarios

def save_jsonl(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
            
if __name__ == "__main__":
    print("[Dataset] Generating 2,000 synthetic disaster scenarios...")
    dataset = generate_scenarios(2000)
    save_jsonl(dataset, OUTPUT_PATH)
    print(f"[Dataset] Success! Saved {len(dataset)} scenarios to {OUTPUT_PATH}")
