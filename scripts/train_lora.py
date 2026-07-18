#!/usr/bin/env python3
"""Training entry point for the Commander adapter.

This intentionally fails with setup guidance when optional training dependencies
are absent. Runtime and benchmark environments do not need the training stack.
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    try:
        import peft  # noqa: F401
        import transformers  # noqa: F401
        import trl  # noqa: F401
    except ImportError as error:
        raise SystemExit(
            "LoRA training requires the optional training environment. "
            "Install requirements-training.txt before running this command."
        ) from error

    raise SystemExit(
        "Training backend is intentionally gated until the dataset and base model "
        f"are confirmed: {args.base_model} -> {args.output_dir}"
    )


if __name__ == "__main__":
    main()
