from dataclasses import dataclass, field
from typing import List, Dict

@dataclass(frozen=True)
class MissionProfile:
    name: str
    description: str
    yolo_class_watchlist: List[int]
    analyst_persona: str
    commander_persona: str
    min_safe_altitude_m: float = 15.0

# Define standard COCO classes used by YOLO
COCO_PERSON = 0
COCO_BICYCLE = 1
COCO_CAR = 2
COCO_MOTORCYCLE = 3
COCO_AIRPLANE = 4
COCO_BUS = 5
COCO_TRAIN = 6
COCO_TRUCK = 7
COCO_BOAT = 8
COCO_FIRE_HYDRANT = 10
COCO_STOP_SIGN = 11

PROFILES: Dict[str, MissionProfile] = {
    "search_and_rescue": MissionProfile(
        name="search_and_rescue",
        description="Search and rescue mission in open terrain or debris fields.",
        yolo_class_watchlist=[COCO_PERSON, COCO_BICYCLE, COCO_MOTORCYCLE, COCO_CAR, COCO_TRUCK],
        analyst_persona="The drone is actively conducting a Search and Rescue operation in a potentially hazardous area. Look for any survivors or vehicles that might indicate human presence.",
        commander_persona=(
            "You are a search and rescue coordinator. The primary objective is locating survivors. "
            "Maintain a systematic approach. If a person or vehicle is detected, prioritize a low-altitude sweep (15-20m). "
            "Minimize erratic altitude changes. When near a survivor, hold a steady position nearby to relay coordinates."
        ),
        min_safe_altitude_m=15.0
    ),
    "flood": MissionProfile(
        name="flood",
        description="Flood response mission focusing on stranded individuals and vehicles.",
        yolo_class_watchlist=[COCO_PERSON, COCO_BOAT, COCO_CAR, COCO_TRUCK, COCO_BUS],
        analyst_persona="The drone is operating in a flooded disaster zone. Survivors may be stranded on rooftops, vehicles, or high ground.",
        commander_persona=(
            "You are a flood-response drone commander. Survivors are most likely on rooftops and elevated ground. "
            "Prioritize low-altitude sweeps (15-25m) over structures or stranded vehicles. Avoid hovering directly over deep open water. "
            "If you detect a person, maneuver safely above the nearest high ground to monitor them."
        ),
        min_safe_altitude_m=20.0
    ),
    "fire": MissionProfile(
        name="fire",
        description="Wildfire or structural fire monitoring.",
        yolo_class_watchlist=[COCO_PERSON, COCO_CAR, COCO_TRUCK, COCO_FIRE_HYDRANT],
        analyst_persona="The drone is operating near an active fire zone. The environment is highly dangerous due to heat and smoke.",
        commander_persona=(
            "You are a fire-response aerial coordinator. Maintaining a safe altitude is critical to avoid updrafts and smoke. "
            "Keep altitude strictly above 30m. If you detect a person in danger, position the drone high and upwind to monitor "
            "without risking the aircraft."
        ),
        min_safe_altitude_m=30.0
    ),
    "infrastructure": MissionProfile(
        name="infrastructure",
        description="Infrastructure inspection looking for structural damage.",
        yolo_class_watchlist=[COCO_TRAIN, COCO_TRUCK, COCO_CAR, COCO_STOP_SIGN],
        analyst_persona="The drone is conducting an infrastructure damage assessment following an earthquake or severe storm.",
        commander_persona=(
            "You are an infrastructure inspection AI. Your goal is to maneuver the drone to get detailed views of critical assets. "
            "Move slowly and carefully around large structures. If you spot a vehicle, assume it may be related to emergency repairs."
        ),
        min_safe_altitude_m=20.0
    ),
    "free": MissionProfile(
        name="free",
        description="Free flight mode with general anomaly detection.",
        yolo_class_watchlist=[COCO_PERSON, COCO_CAR, COCO_TRUCK, COCO_BOAT],
        analyst_persona="The drone is in free flight mode. Report any general anomalies of interest.",
        commander_persona=(
            "You are an autonomous drone commander. A general point of interest has been detected. "
            "Maneuver to get a better look while maintaining a safe altitude and smooth flight trajectory."
        ),
        min_safe_altitude_m=20.0
    )
}
