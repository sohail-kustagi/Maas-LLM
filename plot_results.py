import matplotlib.pyplot as plt
import numpy as np
import os

# Set style
plt.style.use('ggplot')

# 1. LLM Reasoning Latency Plot
labels = ['Baseline (No LoRA)', 'Fine-Tuned (LoRA)']
x86_latency = [12.96, 24.36]
graviton_latency = [10.21, 25.03]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - width/2, x86_latency, width, label='Local x86 CPU', color='#E24A33')
rects2 = ax.bar(x + width/2, graviton_latency, width, label='AWS Graviton (ARM NEON)', color='#348ABD')

ax.set_ylabel('Latency per command (seconds)')
ax.set_title('LLM Reasoning Latency: x86 vs AWS Graviton')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Add values on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}s',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('latency_chart.png', dpi=300)
plt.close()

# 2. Vision Speed (FPS) Plot
x86_fps = [9.06, 9.42]
graviton_fps = [5.27, 6.67]

fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - width/2, x86_fps, width, label='Local x86 CPU', color='#E24A33')
rects2 = ax.bar(x + width/2, graviton_fps, width, label='AWS Graviton (ARM NEON)', color='#348ABD')

ax.set_ylabel('Frames Per Second (FPS)')
ax.set_title('YOLO Vision Speed: x86 vs AWS Graviton')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('fps_chart.png', dpi=300)
plt.close()

# 3. CPU Hotspots Plot
hotspot_labels = [
    'ggml_gemv_q4_K_8x8_q8_K',
    'ggml_gemv_q5_K_8x8_q8_K',
    'ggml_gemv_q6_K_8x8_q8_K',
    'ggml_vec_dot_f16',
    'ggml_gemm_q4_K_8x8_q8_K',
    'jit_sve_conv_fwd_kernel'
]
hotspot_times = [10290, 8080, 5878, 5296, 3963, 1150]

fig, ax = plt.subplots(figsize=(10, 6))
y_pos = np.arange(len(hotspot_labels))
ax.barh(y_pos, hotspot_times, align='center', color='#988ED5')
ax.set_yticks(y_pos)
ax.set_yticklabels(hotspot_labels)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('CPU Time (ms)')
ax.set_title('Top Execution Hotspots on AWS Graviton (Arm Performix)')

for i, v in enumerate(hotspot_times):
    ax.text(v + 100, i + 0.1, str(v), color='black', fontweight='bold')

plt.tight_layout()
plt.savefig('hotspots_chart.png', dpi=300)
plt.close()

print("Charts successfully generated.")
