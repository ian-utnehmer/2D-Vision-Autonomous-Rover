#!/usr/bin/env python3
import jetson_inference
import jetson_utils
import numpy as np
import threading
import sys
import time

# --- TRAINED COEFFICIENTS
A = 0.3045211441
B = -2.2437643455
C = 5.6336667304
D = -4.3246513581

# --- CONFIG ---
SEG_MODEL = "/home/jetson/paper_project/models/paper_resnet18_fpn/paper_resnet18_unet.onnx"
LABELS = "/home/jetson/paper_project/models/paper_resnet18_fpn/labels.txt"
PAPER_CLASS_ID = 1
MIN_PIXELS = 250

# --- GLOBALS ---
latest_depth_val = 0.0
pixel_count = 0
running = True

# --- INIT ---
net = jetson_inference.segNet(argv=[f'--model={SEG_MODEL}', f'--labels={LABELS}'])
depth_net = jetson_inference.depthNet(argv=['--model=monodepth-resnet18'])
camera = jetson_utils.videoSource("csi://0",
                                  argv=['--input-width=640', '--input-height=360', '--input-flip=rotate-180'])
display = jetson_utils.videoOutput("webrtc://@:8554/output")

seg_w, seg_h = net.GetGridWidth(), net.GetGridHeight()
class_mask = jetson_utils.cudaAllocMapped(width=seg_w, height=seg_h, format='gray8')


def camera_loop():
    global latest_depth_val, pixel_count, running
    while running:
        img = camera.Capture()
        if img is None: continue

        net.Process(img)
        net.Mask(class_mask, filter_mode='point')
        mask_array = np.squeeze(jetson_utils.cudaToNumpy(class_mask))

        depth_net.Process(img)
        depth_field = depth_net.GetDepthField()
        depth_array = np.squeeze(jetson_utils.cudaToNumpy(depth_field))

        y_seg, x_seg = np.where(mask_array == PAPER_CLASS_ID)
        pixel_count = len(y_seg)

        if pixel_count > MIN_PIXELS:
            depth_h, depth_w = depth_array.shape
            x_depth = np.clip((x_seg * depth_w / seg_w).astype(np.int32), 0, depth_w - 1)
            y_depth = np.clip((y_seg * depth_h / seg_h).astype(np.int32), 0, depth_h - 1)
            latest_depth_val = float(np.mean(depth_array[y_depth, x_depth]))

        net.Overlay(img, filter_mode='linear')
        display.Render(img)


# Start background camera thread
thread = threading.Thread(target=camera_loop)
thread.daemon = True
thread.start()

print("\n" + "=" * 50)
print("  LIVE DISTANCE VALIDATION")
print("=" * 50)
print("Move the paper to check accuracy against a tape measure.")
print("Press Ctrl+C to stop.")
print("=" * 50 + "\n")

try:
    while running:
        if pixel_count > MIN_PIXELS:
            # The Calibration Formula: A*x^3 + B*x^2 + C*x + D
            x = latest_depth_val
            dist_m = (A * x ** 3) + (B * x ** 2) + (C * x) + D

            # Clip negative results if any
            dist_m = max(0, dist_m)
            dist_in = dist_m * 39.37

            print(f"\r[LOCKED] Raw: {x:.4f} | Meters: {dist_m:.3f}m | Inches: {dist_in:.1f}\"    ", end='', flush=True)
        else:
            print(f"\r[SEARCHING...] Pixels: {pixel_count}                          ", end='', flush=True)

        time.sleep(0.1)

except KeyboardInterrupt:
    running = False

print("\nCleaning up and exiting.")