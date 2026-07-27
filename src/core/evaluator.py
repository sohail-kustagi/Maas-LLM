import time
import json
import os

class HackathonEvaluator:
    def __init__(self, video_file: str, profile_name: str):
        self.video_file = video_file
        self.profile_name = profile_name
        self.start_time = time.time()
        
        # Vision Metrics
        self.total_frames_processed = 0
        self.total_anomalies_detected = 0
        self.confidence_sum = 0.0
        
        # LLM Metrics
        self.total_llm_calls = 0
        self.total_tokens_generated = 0
        self.total_llm_latency = 0.0
        self.valid_json_outputs = 0
        self.invalid_json_outputs = 0
        
    def log_frame(self):
        self.total_frames_processed += 1
        
    def log_detection(self, anomaly_type: str, confidence: float):
        self.total_anomalies_detected += 1
        self.confidence_sum += confidence
        
    def log_llm_generation(self, tokens: int, latency: float, valid_json: bool):
        self.total_llm_calls += 1
        self.total_tokens_generated += tokens
        self.total_llm_latency += latency
        if valid_json:
            self.valid_json_outputs += 1
        else:
            self.invalid_json_outputs += 1

    def generate_report(self, output_path: str = "hackathon_report.md"):
        total_time = time.time() - self.start_time
        fps = self.total_frames_processed / total_time if total_time > 0 else 0
        
        avg_conf = (self.confidence_sum / self.total_anomalies_detected) * 100 if self.total_anomalies_detected > 0 else 0
        avg_latency = self.total_llm_latency / self.total_llm_calls if self.total_llm_calls > 0 else 0
        avg_tps = self.total_tokens_generated / self.total_llm_latency if self.total_llm_latency > 0 else 0
        
        json_accuracy = 0
        if self.total_llm_calls > 0:
            json_accuracy = (self.valid_json_outputs / self.total_llm_calls) * 100

        report = f"""# MAAS-LLM: Hackathon Performance Evaluation

## Test Configuration
- **Video Source:** `{self.video_file}`
- **Mission Profile:** `{self.profile_name}`
- **Total Duration:** `{total_time:.2f} seconds`

## 👁️ Edge Vision (YOLO) Metrics
- **Frames Processed:** {self.total_frames_processed} ({fps:.2f} FPS)
- **Anomalies Detected:** {self.total_anomalies_detected}
- **Average Detection Confidence:** {avg_conf:.1f}%

## 🧠 LLM Commander (Phi-3) Metrics
- **Total Commands Generated:** {self.total_llm_calls}
- **Average Latency:** {avg_latency:.2f}s per command
- **Inference Speed:** {avg_tps:.2f} Tokens/sec
- **Valid JSON formatting rate:** {json_accuracy:.1f}%

---
*Generated automatically by MAAS-LLM Evaluator for the Arm AI Optimization Challenge.*
"""
        
        with open(output_path, "w") as f:
            f.write(report)
        print(f"\n[Evaluator] 🏆 Hackathon report generated at {output_path}!")
