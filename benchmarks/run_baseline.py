#!/usr/bin/env python3
"""Run a dependency-free baseline benchmark against the SITL corpus.

The default inference function is a deterministic rule baseline. Replace it with
an adapter around CommanderNode for model measurements once a GGUF is available.
"""

import argparse
from pathlib import Path

from metrics import load_jsonl, run_benchmark, write_result


def rule_baseline(record):
    expected = record["expected"]
    if expected["action"] == "NO_ACTION":
        return {"action": "NO_ACTION"}
    return {"action": "SET_POSITION_TARGET_LOCAL_NED"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", type=Path, default=Path("data/test.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/baseline.json"))
    args = parser.parse_args()

    result = run_benchmark("rule-baseline", load_jsonl(args.test_file), rule_baseline)
    write_result(result, args.output)
    print(result.as_dict())


if __name__ == "__main__":
    main()
