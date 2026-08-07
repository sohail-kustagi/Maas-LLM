import cv2
import numpy as np
from typing import Optional, Dict

class YoloOverlayRenderer:
    """
    Renders bounding boxes, class labels, and a telemetry HUD overlay on OpenCV frames.
    """
    
    def __init__(self, window_name: str = "MAAS-LLM — Live Detection"):
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self._is_active = True

    def render(self, frame: np.ndarray, results, telemetry: Optional[Dict], mission_profile: str, class_map: Dict[int, str]):
        if not self._is_active:
            return

        # 1. Plot YOLO results manually to enforce correct class names from model.names
        annotated_frame = frame.copy()
        if len(results) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                
                # Dynamically pull the class name from the model's internal dictionary
                class_name = results[0].names[class_id]
                label = f"{class_name} {conf:.2f}"
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw text label background and text
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # 2. Add Telemetry HUD Overlay
        if telemetry:
            hud_lines = [
                f"Mode: {telemetry.get('flight_mode', 'UNKNOWN')}",
                f"Alt: {telemetry.get('alt', 0):.1f}m",
                f"Lat: {telemetry.get('lat', 0):.5f}",
                f"Lon: {telemetry.get('lon', 0):.5f}",
                f"Batt: {telemetry.get('battery_percent', 0)}%"
            ]
            
            y0, dy = 30, 30
            for i, line in enumerate(hud_lines):
                y = y0 + i * dy
                cv2.putText(annotated_frame, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                cv2.putText(annotated_frame, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 3. Add Mission Profile Badge
        badge_text = f"PROFILE: {mission_profile.upper()}"
        text_size, _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        text_w, text_h = text_size
        
        # Position top right
        img_h, img_w, _ = annotated_frame.shape
        x_badge = img_w - text_w - 20
        y_badge = 30
        
        cv2.rectangle(annotated_frame, (x_badge - 10, y_badge - text_h - 10), (x_badge + text_w + 10, y_badge + 10), (0, 0, 255), -1)
        cv2.putText(annotated_frame, badge_text, (x_badge, y_badge), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 4. Display Frame
        cv2.imshow(self.window_name, annotated_frame)
        
        # Pump event loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.stop()

    def stop(self):
        self._is_active = False
        try:
            cv2.destroyWindow(self.window_name)
        except:
            pass
