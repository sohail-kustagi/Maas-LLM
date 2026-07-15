import cv2
import asyncio
from ultralytics import YOLO

class WatchdogNode:
    def __init__(self, model_path="yolov10n.pt"):
        # Load lightweight YOLO model
        print(f"[Watchdog] Loading Vision Model: {model_path}")
        self.model = YOLO(model_path)
        self.cap = None

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
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls == 0 and conf > 0.6:  # 0 is 'person' in standard COCO
                        anomaly_detected = True
                        break

            if anomaly_detected:
                print("[Watchdog] TRIGGER: Anomaly (Person) detected with high confidence!")
                # Here we would normally yield the trigger to Node B (Analyst)
                # But for now we just print it.
                await asyncio.sleep(2) # Prevent spamming triggers every frame

            # Yield to event loop to allow other async tasks to run
            await asyncio.sleep(0.01)

    def stop(self):
        if self.cap:
            self.cap.release()
            print("[Watchdog] Camera released.")
