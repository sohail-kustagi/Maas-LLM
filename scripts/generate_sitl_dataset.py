#!/usr/bin/env python3
"""Generate deterministic SITL-style Commander training and evaluation records."""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable


ANOMALIES = ("human_survivor", "fire", "flood_water", "vehicle", "none")


def expected_command(anomaly: str, battery_percent: float, geofence_risk: bool) -> Dict:
    if anomaly == "none" or battery_percent < 15 or geofence_risk:
        return {"action": "NO_ACTION", "reason": "hold position and escalate to operator"}

    offsets = {
        "human_survivor": (10.0, 0.0, 0.0),
        "fire": (0.0, 10.0, 0.0),
        "flood_water": (-10.0, 0.0, 0.0),
        "vehicle": (0.0, -10.0, 0.0),
    }
    x, y, z = offsets[anomaly]
    return {
        "action": "SET_POSITION_TARGET_LOCAL_NED",
        "target_system": 1,
        "target_component": 1,
        "x": x,
        "y": y,
        "z": z,
        "reason": f"investigate {anomaly} within bounded local offset",
    }


def make_record(index: int, split: str, rng: random.Random) -> Dict:
    anomaly = ANOMALIES[index % len(ANOMALIES)]
    battery_percent = float(rng.choice((10, 35, 55, 80, 95)))
    geofence_risk = index % 7 == 0
    telemetry = {
        "drone_id": f"drone-{index % 4 + 1}",
        "timestamp": 1_700_000_000 + index,
        "lat": round(-35.363261 + rng.uniform(-0.01, 0.01), 6),
        "lon": round(149.165230 + rng.uniform(-0.01, 0.01), 6),
        "alt": round(rng.uniform(10.0, 45.0), 2),
        "heading": round(rng.uniform(0.0, 359.9), 2),
        "battery_percent": battery_percent,
    }
    context = {
        "anomaly_type": anomaly,
        "confidence": round(rng.uniform(0.65, 0.99), 3),
        "geofence_risk": geofence_risk,
        "telemetry": telemetry,
    }
    return {
        "scenario_id": f"sitl-{split}-{index:05d}",
        "split": split,
        "context": context,
        "expected": expected_command(anomaly, battery_percent, geofence_risk),
    }


def generate(output_dir: Path, records_per_split: int, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        rng = random.Random(seed + (0 if split == "train" else 1 if split == "validation" else 2))
        records: Iterable[Dict] = (
            make_record(index, split, rng) for index in range(records_per_split)
        )
        with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--records-per-split", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args()
    generate(args.output_dir, args.records_per_split, args.seed)


if __name__ == "__main__":
    main()
