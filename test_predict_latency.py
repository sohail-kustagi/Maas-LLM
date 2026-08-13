import time
import numpy as np
from ultralytics import YOLO

model = YOLO("weights/best_openvino_model", task="detect")
import openvino as ov
_original_compile = ov.Core.compile_model
def _patched_compile(self, model, device_name=None, config=None, **kwargs):
    if config is None: config = {}
    config["PERFORMANCE_HINT"] = "LATENCY"
    return _original_compile(self, model, "GPU", config, **kwargs)
ov.Core.compile_model = _patched_compile

frame = np.zeros((640, 640, 3), dtype=np.uint8)
print("Warming up...")
model.predict(frame, device="cpu", verbose=False)

start = time.time()
for _ in range(20):
    model.predict(frame, device="cpu", verbose=False)
total = time.time() - start

print(f"20 inferences took {total:.2f}s ({20/total:.2f} fps)")
