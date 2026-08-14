# MAAS-LLM: End-to-End Video Benchmark

## Test Configuration
- **Video:** Pella Home Fire Drone Footage (720p)
- **Environment:** Local PC (x86 CPU Baseline)
- **Vision Model:** YOLOv10 (PyTorch CPU)
- **Reasoning Model:** Phi-3 4k Instruct (llama.cpp)

## Stage 1: Vanilla Baseline (No LoRA)
- **Vision Speed (YOLO):** 8.77 FPS
- **LLM Reasoning Latency:** 26.19s per command

## Stage 2: Fine-Tuned LoRA (Disaster Analyst)
- **Vision Speed (YOLO):** 10.33 FPS
- **LLM Reasoning Latency:** 29.86s per command

---
*Generated automatically by MAAS-LLM Evaluator for the Arm AI Optimization Challenge.*
