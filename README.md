# MAAS AI Pipeline

This directory contains the AI pipeline for the MAAS hackathon. The frontend is out of scope. The Raspberry Pi remains a lightweight edge device; model inference runs locally during development and on AWS Graviton for the official ARM64 measurements.

## Phase Status

1. **Contract and safety:** typed telemetry, vision, and command objects plus Commander validation.
2. **Pipeline baseline:** async Watchdog queue, typed telemetry adapter, Analyst context, and validated Commander output.
3. **SITL data:** deterministic scenario generator with train, validation, and test splits.
4. **Baseline measurement:** benchmark result format and dependency-free baseline runner.
5. **Commander adaptation:** optional training requirements and a gated LoRA entry point.
6. **Node optimization:** Watchdog event sampling and Analyst event gating boundary.
7. **AWS packaging:** ARM64 Dockerfile, runtime dependency split, and model deployment script.
8. **ARM benchmarking:** environment-aware benchmark entry point ready for Graviton and Performix evidence.

The adaptation script is intentionally gated until a base model and training environment are explicitly selected. The current benchmark adapter is deterministic scaffolding, not a model-performance result.

## Local Checks

From this directory:

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -v
python3 scripts/generate_sitl_dataset.py --output-dir data --records-per-split 100
python3 scripts/validate_dataset.py data/train.jsonl data/validation.jsonl data/test.jsonl
PYTHONPATH=. python3 benchmarks/run_baseline.py
```

The sandbox runtime accepts either explicit JSON telemetry or raw MAVLink datagrams from the edge bridge. Raw MAVLink decoding requires the `pymavlink` runtime dependency and uses `HEARTBEAT`, `GLOBAL_POSITION_INT`, and `SYS_STATUS` messages.

For local Watchdog testing, the built-in camera is used by default:

```bash
export CAMERA_INDEX=0
export YOLO_MODEL_PATH=yolov10n.pt
export VISION_CONFIDENCE=0.6
export VISION_SAMPLE_INTERVAL=0.2
```

OpenCV has confirmed that camera index `0` returns `640x480` frames on the current development machine. Camera input validates capture and detection plumbing, but aerial accuracy still requires drone imagery or a labeled aerial dataset.

For the current local SITL setup, set `SITL_PORT=5762` when starting the edge bridge if that is the port emitting telemetry. The bridge forwards the raw stream to `127.0.0.1:9000` when `SANDBOX_MODE=true`.

The JSON adapter format is:

```json
{
  "drone_id": "drone-1",
  "timestamp": 1700000000,
  "lat": -35.363261,
  "lon": 149.165230,
  "alt": 20.0,
  "heading": 90.0,
  "battery_percent": 80.0
}
```

Malformed telemetry is rejected and logged. Raw MAVLink bytes are decoded at the Python boundary rather than silently treated as JSON.

## AWS Graviton

Use a fresh ARM64 Ubuntu instance and pin the model artifact before running the deployment script:

```bash
export MODEL_URL='https://your-pinned-model-artifact-url'
./scripts/deploy_graviton.sh
PYTHONPATH=. python3 scripts/run_graviton_benchmark.py
```

Build the ARM64 image with:

```bash
docker buildx build --platform linux/arm64 -f Dockerfile.arm64 -t maas-llm:arm64 .
```

Record the instance type, region, OS, kernel, compiler, `llama.cpp` version, model hash, thread settings, warm/cold state, and raw benchmark output. Run Arm Performix against the same model and scenario set. Never label the local x86 result as Graviton evidence.

## Model Artifacts

Do not commit `.gguf`, `.pt`, virtual environments, credentials, or large generated artifacts. Keep model revisions, dataset versions, schemas, and benchmark metadata together when publishing a reproducible experiment.
