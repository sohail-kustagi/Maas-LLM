FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv build-essential cmake \
    libgl1 libglib2.0-0 ffmpeg wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install basic Python dependencies
# Standard pip install on Arm64 pulls the correct PyTorch wheels natively
RUN python3 -m pip install torch torchvision torchaudio \
    ultralytics opencv-python-headless websockets pydantic fastapi uvicorn python-multipart

# Copy patches and scripts for Graviton (SVE NEON) compilation
COPY patch_sve.py ./

# Download and compile llama-cpp-python for ARM NEON
RUN pip download --no-deps llama-cpp-python==0.3.34 && \
    tar -xzf llama_cpp_python-0.3.34.tar.gz && \
    python3 patch_sve.py llama_cpp_python-0.3.34/vendor/llama.cpp/ggml/src/ggml-cpu/arch/arm/repack.cpp && \
    cd llama_cpp_python-0.3.34 && \
    CMAKE_ARGS="-DGGML_NATIVE=ON" CFLAGS="-mcpu=neoverse-v1" CXXFLAGS="-mcpu=neoverse-v1" pip install --no-cache-dir --force-reinstall . && \
    cd .. && rm -rf llama_cpp_python-0.3.34*

# Copy the rest of the application code
COPY . /app

EXPOSE 8000

# By default, start the VOD API (docker-compose will override this for the Live Agent)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
