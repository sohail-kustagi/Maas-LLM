import cv2, time
from ultralytics import YOLO

cap = cv2.VideoCapture("test.webm")
if not cap.isOpened():
    print("Cannot open test.webm")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("test_out.mp4", fourcc, fps, (w, h))

model = YOLO("weights/best_openvino_model", task="detect")
import openvino as ov
_original_compile = ov.Core.compile_model
def _patched_compile(self, model, device_name=None, config=None, **kwargs):
    if config is None: config = {}
    config["PERFORMANCE_HINT"] = "THROUGHPUT"
    return _original_compile(self, model, "GPU", config, **kwargs)
ov.Core.compile_model = _patched_compile

frame_count = 0
start = time.time()
inf_time = 0
write_time = 0

while True:
    ret, frame = cap.read()
    if not ret or frame_count > 100: break
    frame_count += 1
    
    if frame_count % 5 == 0:
        t0 = time.time()
        res = model.predict(frame, device="cpu", verbose=False)
        inf_time += time.time() - t0
        
    t0 = time.time()
    out.write(frame)
    write_time += time.time() - t0

out.release()
cap.release()
total = time.time() - start
print(f"Total: {total:.2f}s, Inference: {inf_time:.2f}s, Write: {write_time:.2f}s")
