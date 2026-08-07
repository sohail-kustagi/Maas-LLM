import gradio as gr
import cv2
import os
import json
import asyncio
from ultralytics import YOLO
from src.nodes.analyst import AnalystNode
from src.nodes.commander import CommanderNode
from src.core.types import VisionEvent
from src.core.mission_profiles import PROFILES

# We must run asyncio gracefully inside Gradio
def process_video(video_path, mission_profile_name):
    if not video_path:
        yield None, "Please upload a video."
        return

    # Load Model and Nodes
    print("[WebUI] Loading Pipeline...")
    model = YOLO("weights/best.pt")
    print("=== MODEL CLASS DICTIONARY ===")
    print(model.names)
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
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Run inference
        results = model(frame, verbose=False)
        
        # Plot YOLO results manually to enforce correct class names from model.names
        annotated_frame = frame.copy()
        if len(results) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                
                # Dynamically pull the class name
                class_name = results[0].names[class_id]
                label = f"{class_name} {conf:.2f}"
                
                # Draw bounding box and label
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                (w_txt, h_txt), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + w_txt, y1), (0, 255, 0), -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        out.write(annotated_frame)
        
        # Every 30 frames (~1 sec), trigger the LLM analyst if detections exist
        if frame_count % 30 == 0:
            best_cls = None
            best_conf = 0.0
            
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls in model.names and conf > best_conf:
                    best_cls = cls
                    best_conf = conf
                    
            if best_cls is not None:
                anomaly_type = model.names[best_cls]
                log_output += f"\n--- Frame {frame_count} ---\n[Watchdog] Detected {anomaly_type} (conf: {best_conf:.2f})\n"
                yield out_path, log_output
                
                # Mock Telemetry
                dummy_telemetry = {
                    "lat": 34.0522, "lon": -118.2437, "alt": 25.0, "heading": 90, "battery_percent": 85
                }
                
                # Generate Context and Run LLM synchronously for UI demo
                context = analyst.generate_context(anomaly_type, dummy_telemetry, None, None, mission_profile)
                
                # We need to run the async LLM call in a synchronous wrapper
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    command = loop.run_until_complete(commander.generate_mavlink_command(context, dummy_telemetry, mission_profile))
                    log_output += f"[Commander] LLM Response:\n{json.dumps(command, indent=2)}\n"
                    loop.close()
                except Exception as e:
                    log_output += f"[Commander] LLM Error: {e}\n"
                    
                yield out_path, log_output

    cap.release()
    out.release()
    log_output += "\nProcessing Complete."
    
    # Return the final output video and logs
    yield out_path, log_output

with gr.Blocks(title="MAAS-LLM Live Web UI") as demo:
    gr.Markdown("# MAAS-LLM Disaster Response Dashboard")
    gr.Markdown("Upload a drone video feed to process it through the fine-tuned YOLO model and the LLM Disaster Analyst Node.")
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload Drone Feed")
            profile_dropdown = gr.Dropdown(choices=list(PROFILES.keys()), value="search_and_rescue", label="Mission Profile")
            process_btn = gr.Button("Start Analysis", variant="primary")
            
        with gr.Column():
            video_output = gr.Video(label="Detection Output")
            log_output = gr.Textbox(label="LLM Analyst Process & Logs", lines=15, max_lines=30)
            
    process_btn.click(
        fn=process_video,
        inputs=[video_input, profile_dropdown],
        outputs=[video_output, log_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
