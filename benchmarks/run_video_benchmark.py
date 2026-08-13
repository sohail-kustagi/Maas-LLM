#!/usr/bin/env python3
"""
MAAS-LLM End-to-End Video Benchmark (YOLO + Phi-3)
"""

import sys
import os
import json
import time
import asyncio
import cv2
import shutil
from pathlib import Path

# Fix python path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ultralytics import YOLO
from src.nodes.commander import CommanderNode
from src.core.types import TelemetrySnapshot
from src.core.mission_profiles import PROFILES

VIDEO_PATH = "youtube_raw.mp4"
LORA_PATH = Path("weights/phi3-lora.gguf")
LORA_BACKUP_PATH = Path("weights/phi3-lora.gguf.bak")

async def run_video_stage(stage_name):
    print(f"\n{'='*50}\nStarting {stage_name}\n{'='*50}")
    
    # 1. Load YOLO Vision Model
    print("[Vision] Loading YOLOv10...")
    yolo_model = YOLO("weights/best.pt", task="detect")
    
    # 2. Load Phi-3 Commander
    print("[LLM] Loading CommanderNode...")
    commander = CommanderNode()
    commander.set_evaluator(None)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video Loaded: {total_frames} frames @ {fps} FPS")
    
    profile = PROFILES["search_and_rescue"]
    
    frame_count = 0
    inference_count = 0
    total_vision_time = 0.0
    
    total_llm_latency = 0.0
    llm_triggers = 0
    
    print("\nProcessing Video...")
    start_time = time.time()
    
    while cap.isOpened() and frame_count < 300: # Limit to 300 frames (~10 secs) for benchmark
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        
        # Run Vision Inference every 5 frames
        if frame_count % 5 == 0:
            v_start = time.time()
            results = yolo_model.predict(frame, device="cpu", conf=0.25, verbose=False)
            total_vision_time += (time.time() - v_start)
            inference_count += 1
            
            # Check for Fire detections to trigger LLM
            fire_detected = False
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    class_name = yolo_model.names[cls]
                    if class_name.upper() == "FIRE" or class_name.upper() == "SMOKE":
                        fire_detected = True
                        break
                        
            # Force Trigger LLM for benchmarking purposes (simulate swarm routing)
            if llm_triggers < 3: # Limit to 3 triggers per test
                print(f"[Frame {frame_count}] FORCE TRIGGERING LLM Swarm Commander for Benchmark...")
                class_name = "SIMULATED_FIRE"
                
                telemetry = TelemetrySnapshot(
                    drone_id="TEST", timestamp=time.time(),
                    latitude=34.0, longitude=-118.0, altitude_m=20.0,
                    heading_deg=0.0, battery_percent=100.0
                )
                
                # Mock analyst prompt for the vision detection
                prompt = f"Vision Payload Alert: High-confidence {class_name} detected at current coordinates. Provide immediate MAVLink routing."
                
                llm_start = time.time()
                try:
                    # Pass unique anomaly type to bypass 15s cooldown
                    cmd = await commander.generate_mavlink_command(prompt, telemetry, profile, anomaly_type=f"FIRE_{llm_triggers}")
                    llm_latency = time.time() - llm_start
                    total_llm_latency += llm_latency
                    llm_triggers += 1
                    print(f" -> LLM Response in {llm_latency:.2f}s | Valid JSON: {cmd is not None}")
                except Exception as e:
                    print(f" -> LLM Failed: {e}")
                    
    cap.release()
    total_time = time.time() - start_time
    
    avg_vision_fps = inference_count / total_vision_time if total_vision_time > 0 else 0
    avg_llm_latency = total_llm_latency / llm_triggers if llm_triggers > 0 else 0
    
    print(f"\n[{stage_name} Results]")
    print(f"Vision Processing: {avg_vision_fps:.2f} FPS")
    print(f"LLM Reasoning: {avg_llm_latency:.2f}s per command")
    
    return {
        "vision_fps": avg_vision_fps,
        "llm_latency": avg_llm_latency
    }

async def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found at {VIDEO_PATH}")
        return
        
    # ── STAGE 1: Baseline (No LoRA) ──
    if LORA_PATH.exists():
        shutil.move(str(LORA_PATH), str(LORA_BACKUP_PATH))
        print("Moved LoRA adapter out of the way for Baseline test.")
        
    stage1_results = await run_video_stage("Stage 1: End-to-End Baseline (x86 CPU)")
    
    # ── STAGE 2: Fine-Tuned (With LoRA) ──
    if LORA_BACKUP_PATH.exists():
        shutil.move(str(LORA_BACKUP_PATH), str(LORA_PATH))
        print("\nRestored LoRA adapter for Fine-Tuned test.")
        
    stage2_results = await run_video_stage("Stage 2: End-to-End Fine-Tuned (x86 CPU)")
    
    # Generate Report
    report = f"""# MAAS-LLM: End-to-End Video Benchmark

## Test Configuration
- **Video:** Pella Home Fire Drone Footage (720p)
- **Environment:** Local PC (x86 CPU Baseline)
- **Vision Model:** YOLOv10 (PyTorch CPU)
- **Reasoning Model:** Phi-3 4k Instruct (llama.cpp)

## Stage 1: Vanilla Baseline (No LoRA)
- **Vision Speed (YOLO):** {stage1_results['vision_fps']:.2f} FPS
- **LLM Reasoning Latency:** {stage1_results['llm_latency']:.2f}s per command

## Stage 2: Fine-Tuned LoRA (Disaster Analyst)
- **Vision Speed (YOLO):** {stage2_results['vision_fps']:.2f} FPS
- **LLM Reasoning Latency:** {stage2_results['llm_latency']:.2f}s per command

---
*Generated automatically by MAAS-LLM Evaluator for the Arm AI Optimization Challenge.*
"""
    
    with open("hackathon_report.md", "w") as f:
        f.write(report)
        
    print("\nEnd-to-End Report written to hackathon_report.md")

if __name__ == "__main__":
    asyncio.run(main())
