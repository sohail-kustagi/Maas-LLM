import cv2
import asyncio
import time
from ultralytics import YOLO

try:
    from core.types import VisionEvent
except ImportError:
    from src.core.types import VisionEvent

class WatchdogNode:
    def __init__(
        self,
        model_path="yolov10n.pt",
        event_queue=None,
        sample_interval=0.2,
        confidence_threshold=0.6,
    ):
        # Load lightweight YOLO model
        print(f"[Watchdog] Loading Vision Model: {model_path}")
        self.model = YOLO(model_path)
        self.cap = None
        self.event_queue = event_queue
        self.sample_interval = sample_interval
        self.confidence_threshold = confidence_threshold
        self.last_event_at = 0.0

    def start_camera(self, camera_index=0):
        print(f"[Watchdog] Initializing local webcam stream on index {camera_index}...")
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print("[Watchdog] ERROR: Cannot open webcam.")
            return False
        return True

    async def run_vision_loop(self):
        if not self.cap or not self.cap.isOpened():
            print("[Watchdog] Camera not initialized. Exiting vision loop.")
            return

        print("[Watchdog] Vision loop active. Analyzing frames...")
        
        while True:
            # Read a frame
            ret, frame = self.cap.read()
            if not ret:
                print("[Watchdog] Can't receive frame (stream end?). Exiting ...")
                break
            
            # Run YOLOv10 inference
            results = self.model(frame, verbose=False)
            
            # Temporary logic: Just check if we detect a 'person' (class 0 in COCO) with high confidence
            # In disaster response, this triggers the Analyst
            anomaly_detected = False
            highest_confidence = 0.0
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls == 0 and conf > self.confidence_threshold:
                        anomaly_detected = True
                        highest_confidence = max(highest_confidence, conf)
                        break

            if anomaly_detected:
                print("[Watchdog] TRIGGER: Anomaly (Person) detected with high confidence!")
                now = time.time()
                if self.event_queue is not None and now - self.last_event_at >= self.sample_interval:
                    event = VisionEvent(
                        drone_id="local-camera",
                        timestamp=now,
                        anomaly_type="human_survivor",
                        confidence=highest_confidence,
                    )
                    await self.event_queue.put(event)
                    self.last_event_at = now

            # Yield to event loop to allow other async tasks to run
            await asyncio.sleep(0)

    def stop(self):
        if self.cap:
            self.cap.release()
            print("[Watchdog] Camera released.")
