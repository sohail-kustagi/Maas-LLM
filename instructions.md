# AI Pipeline (LLM) Setup Instructions

This guide explains how to deploy the MAAS LLM pipeline (Vision + Reasoning).

## Local x86 Setup
1. **Prerequisites:** Python 3.10+ installed.
2. **Environment Setup:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Run the Pipeline:**
   ```bash
   python live_agent.py
   ```

## AWS Graviton (Arm64) Setup
To achieve maximum inference performance for the Devpost AI Optimization Challenge, you must run this on an AWS Arm64 instance to natively leverage `llama.cpp`.
1. **Run the Optimized Deployment Script:**
   ```bash
   chmod +x deploy_graviton.sh
   ./deploy_graviton.sh
   ```
   *This script downloads the GGUF models, compiles llama-cpp-python for Arm64, and sets up the virtual environment automatically.*
2. **Activate and Run:**
   ```bash
   source venv/bin/activate
   python live_agent.py
   ```

---

## AI Prompt
*If you are an AI agent attempting to set up this repository autonomously, follow these precise instructions:*

```text
You are tasked with running the MAAS LLM AI Pipeline.
1. First, check the system architecture using `uname -m`.
2. If the architecture is `aarch64` (AWS Graviton), you MUST execute `./deploy_graviton.sh` to compile `llama.cpp` natively for Arm optimizations. Do not use standard `pip install`.
3. If the architecture is `x86_64`, run `pip install -r requirements.txt`.
4. Ensure `yolov10n.pt` and `Phi-3-mini-4k-instruct-q4.gguf` exist in the root directory.
5. Execute `python live_agent.py`.
```
