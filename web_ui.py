import gradio as gr
import cv2
import os
import json
import asyncio
import torch
import time
import concurrent.futures
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, get_prediction
from src.nodes.analyst import AnalystNode
from src.nodes.commander import CommanderNode
from src.core.types import VisionEvent, TelemetrySnapshot
from src.core.mission_profiles import PROFILES

# We must run asyncio gracefully inside Gradio
def process_video(video_path, mission_profile_name, conf_threshold, use_sahi):
    if not video_path:
        yield None, "Please upload a video."
        return

    # Load Model and Nodes
    print("[WebUI] Loading Pipeline...")
    
    # OpenVINO Export Logic
    ov_model_dir = "weights/best_openvino_model"
    if not os.path.exists(ov_model_dir):
        print("[INFO] Exporting model to Intel OpenVINO format for speed...")
        temp_model = YOLO("weights/best.pt")
        temp_model.export(format="openvino")
        
    print("[INFO] Initializing Video Pipeline with Intel OpenVINO on CPU")
    
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=ov_model_dir,
        confidence_threshold=conf_threshold,
        device='cpu'
    )
    
    print("=== MODEL CLASS DICTIONARY ===")
    print(detection_model.category_mapping)
    print("==============================")
    
    analyst = AnalystNode()
    commander = CommanderNode()
    commander.set_evaluator(None) # Initializes the LLM engine in CommanderNode
    mission_profile = PROFILES.get(mission_profile_name, PROFILES["search_and_rescue"])

    cap = cv2.VideoCapture(video_path)
    
    # Get video properties for output writer
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out_path = "output_detection.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    log_output = "Pipeline Started...\n"
    yield None, log_output

    frame_count = 0
    last_boxes = []
    
    last_triggered_times = {}
    cooldown_seconds = 15.0
    last_commands = {}
    
    # Async background executor for LLM to prevent video pipeline stalling
    # STRICTLY max_workers=1 to prevent Llama.cpp C-binding segfaults on concurrent calls
    llm_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    shared_logs = []
    
    pipeline_start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        annotated_frame = frame.copy()
        
        # Inference Branching
        if use_sahi:
            frame_stride = 10
            if frame_count % frame_stride == 0:
                prediction_results = get_sliced_prediction(
                    frame,
                    detection_model,
                    slice_height=512,
                    slice_width=512,
                    overlap_height_ratio=0.2,
                    overlap_width_ratio=0.2
                )
                last_boxes = []
                for obj in prediction_results.object_prediction_list:
                    x1, y1, x2, y2 = map(int, [obj.bbox.minx, obj.bbox.miny, obj.bbox.maxx, obj.bbox.maxy])
                    conf = float(obj.score.value)
                    class_name = obj.category.name
                    last_boxes.append((x1, y1, x2, y2, conf, class_name))
        else:
            frame_stride = 1
            if frame_count % frame_stride == 0:
                prediction_results = get_prediction(frame, detection_model)
                last_boxes = []
                for obj in prediction_results.object_prediction_list:
                    x1, y1, x2, y2 = map(int, [obj.bbox.minx, obj.bbox.miny, obj.bbox.maxx, obj.bbox.maxy])
                    conf = float(obj.score.value)
                    class_name = obj.category.name
                    last_boxes.append((x1, y1, x2, y2, conf, class_name))
        
        # Draw bounding boxes (fresh or reused)
        for box in last_boxes:
            x1, y1, x2, y2, conf, class_name = box
            label = f"{class_name} {conf:.2f}"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (w_txt, h_txt), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            
            # Prevent label from rendering off the top edge of the video frame
            if y1 - 20 < 0:
                y_bg = y1
                y_txt = y1 + 15
            else:
                y_bg = y1 - 20
                y_txt = y1 - 5
                
            cv2.rectangle(annotated_frame, (x1, y_bg), (x1 + w_txt, y_bg + 20), (0, 255, 0), -1)
            cv2.putText(annotated_frame, label, (x1, y_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        out.write(annotated_frame)
        
        # Flush background LLM logs to the UI without blocking
        if shared_logs:
            log_output += "".join(shared_logs)
            shared_logs.clear()
            yield out_path, log_output
        
        # Every 30 frames (~1 sec), trigger the LLM analyst if detections exist
        if frame_count % 30 == 0:
            best_cls_name = None
            best_conf = 0.0
            
            for box in last_boxes:
                _, _, _, _, conf, class_name = box
                if conf > best_conf:
                    best_cls_name = class_name
                    best_conf = conf
                    
            if best_cls_name is not None:
                anomaly_type = best_cls_name
                current_video_timestamp = frame_count / fps
                
                # Check Debounce Timer (Per Anomaly Class)
                last_time = last_triggered_times.get(anomaly_type, -999.0)
                if (current_video_timestamp - last_time) >= cooldown_seconds:
                    last_triggered_times[anomaly_type] = current_video_timestamp
                    
                    log_output += f"\n--- Frame {frame_count} ---\n[Watchdog] Detected {anomaly_type} (conf: {best_conf:.2f})\n"
                    yield out_path, log_output
                    
                    # Mock Telemetry
                    dummy_telemetry = TelemetrySnapshot(
                        drone_id="UI_MOCK",
                        timestamp=time.time(),
                        latitude=34.0522,
                        longitude=-118.2437,
                        altitude_m=25.0,
                        heading_deg=90.0,
                        battery_percent=85.0
                    )
                    
                    # Generate Context
                    context = analyst.generate_context(anomaly_type, dummy_telemetry, None, None, mission_profile)
                    
                    # Fire-and-forget background LLM task so video keeps playing at 30fps
                    def run_llm_task(ctx, tel, profile, a_type):
                        try:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            cmd = loop.run_until_complete(commander.generate_mavlink_command(ctx, tel, profile))
                            if cmd:
                                last_commands[a_type] = cmd
                                shared_logs.append(f"\n[Commander] Async Response Ready for {a_type}:\n{json.dumps(cmd, indent=2)}\n")
                            loop.close()
                        except Exception as e:
                            shared_logs.append(f"\n[Commander] Async LLM Error: {e}\n")
                            
                    llm_executor.submit(run_llm_task, context, dummy_telemetry, mission_profile, anomaly_type)
                    
                else:
                    # Within cooldown window, yield previous command without triggering LLM
                    cmd = last_commands.get(anomaly_type)
                    if cmd:
                        log_output += f"\n--- Frame {frame_count} ---\n[Watchdog] Cooldown Active. Reusing previous {anomaly_type} command.\n"
                        yield out_path, log_output

    cap.release()
    out.release()
    llm_executor.shutdown(wait=False)
    
    if shared_logs:
        log_output += "".join(shared_logs)
        
    pipeline_elapsed_time = time.time() - pipeline_start_time
    log_output += f"\nProcessing Complete. Time elapsed: {pipeline_elapsed_time:.2f} seconds."
    
    # Return the final output video and logs
    yield out_path, log_output

with gr.Blocks(title="MAAS-LLM Live Web UI") as demo:
    gr.Markdown("# MAAS-LLM Disaster Response Dashboard")
    gr.Markdown("Upload a drone video feed to process it through the fine-tuned YOLO model and the LLM Disaster Analyst Node.")
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload Drone Feed")
            profile_dropdown = gr.Dropdown(choices=list(PROFILES.keys()), value="search_and_rescue", label="Mission Profile")
            conf_slider = gr.Slider(minimum=0.10, maximum=0.90, value=0.40, step=0.05, label="Confidence Threshold")
            use_sahi_checkbox = gr.Checkbox(label="Enable SAHI (Deep Inspection)", value=False)
            process_btn = gr.Button("Start Analysis", variant="primary")
            
        with gr.Column():
            video_output = gr.File(label="Download Processed Video")
            log_output = gr.Textbox(label="LLM Analyst Process & Logs", lines=15, max_lines=30)
            
    process_btn.click(
        fn=process_video,
        inputs=[video_input, profile_dropdown, conf_slider, use_sahi_checkbox],
        outputs=[video_output, log_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
