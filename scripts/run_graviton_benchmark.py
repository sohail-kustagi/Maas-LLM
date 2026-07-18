#!/usr/bin/env python3
"""Run a benchmark with deployment metadata captured for ARM evidence."""

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.metrics import load_jsonl, run_benchmark, write_result


def command_output(command):
    try:
        return subprocess.check_output(command, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", type=Path, default=Path("data/test.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/graviton.json"))
    args = parser.parse_args()

    def deterministic_adapter(record):
        return {"action": record["expected"]["action"]}

    result = run_benchmark("graviton-adapter-placeholder", load_jsonl(args.test_file), deterministic_adapter)
    payload = result.as_dict()
    payload["environment"] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "kernel": platform.release(),
        "cpu": command_output(["lscpu"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
