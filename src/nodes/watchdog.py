import cv2
import asyncio
import time
from typing import Optional
from ultralytics import YOLO

try:
    from core.types import VisionEvent
    from core.yolo_ui import YoloOverlayRenderer
except ImportError:
    from src.core.types import VisionEvent
    from src.core.yolo_ui import YoloOverlayRenderer


# Default COCO class ID → disaster anomaly type mapping.
# Add custom-trained YOLO class IDs here to extend detection.
DEFAULT_CLASS_MAP: dict[int, str] = {
    0: "human_survivor",    # COCO class 0 = person
    # Example for a custom-trained model:
    # 80: "fire",
    # 81: "flood_water",
    # 82: "vehicle",
}


class WatchdogNode:
    def __init__(
        self,
        model_path: str = "yolov10n.pt",
        event_queue: Optional[asyncio.Queue] = None,
        sample_interval: float = 0.2,
        confidence_threshold: float = 0.6,
        class_map: Optional[dict[int, str]] = None,
        drone_id: str = "local-camera",
        show_ui: bool = False,
        mission_profile_name: str = "sandbox",
    ):
        print(f"[Watchdog] Loading Vision Model: {model_path}")
        self.model = YOLO(model_path)
        self.cap: Optional[cv2.VideoCapture] = None
        self.event_queue = event_queue
        self.sample_interval = sample_interval
        self.confidence_threshold = confidence_threshold
        self.class_map = class_map if class_map is not None else DEFAULT_CLASS_MAP
        self.drone_id = drone_id
        self.show_ui = show_ui
        self.mission_profile_name = mission_profile_name
        self.ui_renderer = YoloOverlayRenderer() if show_ui else None
        self.last_event_at: float = 0.0
        self.evaluator = None

    def set_evaluator(self, evaluator):
        self.evaluator = evaluator

    def start_camera(self, video_source=0) -> bool:
        print(f"[Watchdog] Initializing vision stream on {video_source}...")
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            print("[Watchdog] ERROR: Cannot open camera.")
            return False
        print("[Watchdog] Camera ready.")
        return True

    async def _camera_reader(self) -> None:
        """Continuously reads frames to keep the hardware buffer empty and fresh."""
        while self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.latest_frame = frame
                if self.evaluator:
                    self.evaluator.log_frame()
            else:
                # End of video file
                print("[Watchdog] End of video stream reached.")
                self.latest_frame = None
                if self.cap:
                    self.cap.release()
                break
            # Add a small delay matching 30fps for realistic video playback
            await asyncio.sleep(0.033)

    async def run_vision_loop(self) -> None:
        if not self.cap or not self.cap.isOpened():
            print("[Watchdog] Camera not initialized. Exiting vision loop.")
            return

        print("[Watchdog] Vision loop active. Analyzing frames...")
        
        self.latest_frame = None
        loop = asyncio.get_running_loop()
        cam_task = asyncio.create_task(self._camera_reader())

        try:
            while True:
                frame = self.latest_frame
                if frame is None:
                    if self.cap is None or not self.cap.isOpened():
                        print("[Watchdog] Camera closed. Exiting vision loop.")
                        break
                    await asyncio.sleep(0.1)
                    continue

                # Run YOLO inference in an executor to avoid blocking the main asyncio/UI thread
                def _infer():
                    return self.model(frame, verbose=False)
                
                results = await loop.run_in_executor(None, _infer)

                # Find the best detection across ALL boxes in ALL results
                best_cls: Optional[int] = None
                best_conf: float = 0.0

                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        # Only consider classes we care about and above threshold
                        if cls in self.class_map and conf >= self.confidence_threshold:
                            if conf > best_conf:
                                best_cls = cls
                                best_conf = conf

                if best_cls is not None:
                    now = time.time()
                    if (
                        self.event_queue is not None
                        and now - self.last_event_at >= self.sample_interval
                    ):
                        anomaly_type = self.class_map[best_cls]
                        print(f"[Watchdog] TRIGGER: {anomaly_type} detected (conf={best_conf:.2f})")
                        if self.evaluator:
                            self.evaluator.log_detection(anomaly_type, best_conf)
                        event = VisionEvent(
                            drone_id=self.drone_id,
                            timestamp=now,
                            anomaly_type=anomaly_type,
                            confidence=best_conf,
                        )
                        try:
                            self.event_queue.put_nowait(event)
                        except asyncio.QueueFull:
                            print(f"[Watchdog] Queue full (LLM busy)! Dropping event: {anomaly_type}")
                        self.last_event_at = now

                if self.show_ui and self.ui_renderer and self.ui_renderer._is_active:
                    import src.main
                    current_tel = src.main.current_telemetry
                    self.ui_renderer.render(frame, results, current_tel, self.mission_profile_name, self.class_map)

                import gc
                gc.collect()

                # Yield to event loop
                await asyncio.sleep(0.05)
        finally:
            cam_task.cancel()

    def stop(self) -> None:
        if self.cap:
            self.cap.release()
            print("[Watchdog] Camera released.")
        if self.ui_renderer:
            self.ui_renderer.stop()
