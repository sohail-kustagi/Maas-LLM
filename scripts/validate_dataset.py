#!/usr/bin/env python3
"""Validate generated SITL records before using them for adaptation."""

import argparse
import json
from pathlib import Path

REQUIRED_CONTEXT = {"anomaly_type", "confidence", "geofence_risk", "telemetry"}
REQUIRED_TELEMETRY = {"drone_id", "timestamp", "lat", "lon", "alt", "heading", "battery_percent"}


def validate(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            if not REQUIRED_CONTEXT.issubset(record["context"]):
                raise ValueError(f"{path}:{line_number}: incomplete context")
            if not REQUIRED_TELEMETRY.issubset(record["context"]["telemetry"]):
                raise ValueError(f"{path}:{line_number}: incomplete telemetry")
            if "action" not in record["expected"]:
                raise ValueError(f"{path}:{line_number}: missing expected action")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    total = sum(validate(path) for path in args.files)
    print(f"validated {total} records")


if __name__ == "__main__":
    main()
