#!/bin/bash
# ==============================================================================
# MAAS-LLM: AWS Graviton (Arm64) Deployment & Setup Script
# ==============================================================================
# Run this script on your AWS Graviton EC2 instance (e.g., c7g.xlarge / t4g.xlarge)
# It compiles llama.cpp from source to leverage ARM NEON vector instructions!

set -e

echo "=================================================="
echo "🚀 Starting MAAS-LLM Graviton Setup..."
echo "=================================================="

echo "[1/5] Installing System Dependencies..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv build-essential cmake \
    libgl1 libglib2.0-0 ffmpeg wget

echo "[2/5] Creating Python Virtual Environment..."
python3 -m venv venv
source venv/bin/activate

echo "[3/5] Installing Vision & Core Dependencies..."
pip install --upgrade pip
# Standard pip install on Arm64 will pull the correct PyTorch wheels
pip install torch torchvision torchaudio
pip install ultralytics opencv-python-headless websockets pydantic

echo "[4/5] Compiling LLAMA.CPP for ARM NEON..."
# This is the secret sauce for the hackathon! 
# We force CMAKE to compile natively for the Graviton chip, unlocking ARM NEON math.
pip download --no-deps llama-cpp-python==0.3.34
tar -xzf llama_cpp_python-0.3.34.tar.gz
python3 patch_sve.py llama_cpp_python-0.3.34/vendor/llama.cpp/ggml/src/ggml-cpu/arch/arm/repack.cpp
cd llama_cpp_python-0.3.34
CMAKE_ARGS="-DGGML_NATIVE=ON" CFLAGS="-mcpu=neoverse-v1" CXXFLAGS="-mcpu=neoverse-v1" pip install --no-cache-dir --force-reinstall .
cd ..

echo "=================================================="
echo "✅ Setup Complete!"
echo "=================================================="
echo ""
echo "Next Steps:"
echo "1. Upload your 'weights/' folder containing 'best.pt' and 'phi3-lora.gguf' to this server."
echo "2. Upload your drone video to this server."
echo "3. Run the benchmark: source venv/bin/activate && python benchmarks/run_video_benchmark.py"
echo "=================================================="
