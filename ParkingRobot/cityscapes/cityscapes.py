#!/usr/bin/env python3
import jetson_inference
import jetson_utils
import numpy as np

# --- Configuration ---
JETSON_IP = "<JETSON_LOCAL_IP>"

print("[INFO] Loading Full Cityscapes Model...")


# ResNet18 for speed as the Jetson Nano is low on compute
# We previously trained a model on ResNet50 for paper detection but found
# the output was too slow (1-3Hz).
net = jetson_inference.segNet("fcn-resnet18-cityscapes")

# Print labels
print(f"[INFO] Network loaded with {net.GetNumClasses()} classes:")
for n in range(net.GetNumClasses()):
    print(f"ID {n}: {net.GetClassDesc(n)}")

net.SetOverlayAlpha(150)

# Setup Camera
camera = jetson_utils.videoSource("csi://0", argv=[
    '--input-width=1280', '--input-height=720', '--input-rate=10', '--input-flip=rotate-180'
])

# Setup WebRTC Output 
display = jetson_utils.videoOutput("webrtc://@:8554/output", argv=['--headless'])

# Mask Buffers
grid_width, grid_height = net.GetGridSize()
class_mask = jetson_utils.cudaAllocMapped(width=grid_width, height=grid_height, format='gray8')

print(f"[INFO] Full Cityscapes Active! View at http://{JETSON_IP}:8554")

try:
    while True:
        img = camera.Capture(timeout=1000)
        if img is None: continue

        # Run inference
        net.Process(img)

        # Draw the colorized overlay
        net.Overlay(img, filter_mode='linear')

        display.Render(img)

        if not camera.IsStreaming() or not display.IsStreaming():
            break

except Exception as e:
    print(f"[ERROR] {e}")
finally:
    print("\n[INFO] Shutting down...")
