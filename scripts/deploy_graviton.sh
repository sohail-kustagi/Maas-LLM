#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-$PWD/models}"
MODEL_FILE="${MODEL_FILE:-commander-q4.gguf}"
MODEL_URL="${MODEL_URL:?Set MODEL_URL to a pinned model artifact URL}"

mkdir -p "$MODEL_DIR"
if [[ ! -f "$MODEL_DIR/$MODEL_FILE" ]]; then
  curl --fail --location --retry 3 --output "$MODEL_DIR/$MODEL_FILE" "$MODEL_URL"
fi

printf 'Architecture: '
uname -m
printf 'Model: %s\n' "$MODEL_DIR/$MODEL_FILE"
printf 'Runtime image build: docker buildx build --platform linux/arm64 -f Dockerfile.arm64 -t maas-llm:arm64 .\n'
printf 'Run the benchmark after installing the pinned runtime dependencies and dataset.\n'
