#!/usr/bin/env python3
import jetson_inference
import jetson_utils
import numpy as np
import threading
import time
import sys

# --- CONFIG ---
MODEL_NAME = "fcn-resnet18-cityscapes"
MIN_PIXELS = 500  # Adjust based on how much grass wanted in frame before locking

# --- CALIBRATION
A, B, C, D = 0.3045211441, -2.2437643455, 5.6336667304, -4.3246513581

# --- GLOBALS ---
latest_depth_val = 0.0
pixel_count = 0
running = True
target_id = -1
target_name = ""

# --- INIT ---
net = jetson_inference.segNet(MODEL_NAME)
depth_net = jetson_inference.depthNet(argv=['--model=monodepth-resnet18'])

# AUTO-DETECT VEGETATION ID
for n in range(net.GetNumClasses()):
    desc = net.GetClassDesc(n).lower()
    if 'vegetation' in desc or 'grass' in desc:
        target_id = n
        target_name = desc
        print(f"[INFO] Found target class '{desc}' at ID: {target_id}")
        break

if target_id == -1:
    print("[ERROR] Could not find vegetation/grass in the model labels!")
    sys.exit(1)

net.SetOverlayAlpha(150.0)

# Camera and Display (Native for NoMachine)
camera = jetson_utils.videoSource("csi://0", argv=[
    '--input-width=1280', '--input-height=720', '--input-rate=15', '--input-flip=rotate-180'
])
display = jetson_utils.videoOutput("display://0")

# Buffers
grid_w, grid_h = net.GetGridWidth(), net.GetGridHeight()
class_mask = jetson_utils.cudaAllocMapped(width=grid_w, height=grid_h, format='gray8')
font = jetson_utils.cudaFont()

def camera_loop():
    global latest_depth_val, pixel_count, running
    while running:
        img = camera.Capture()
        if img is None: continue

        # Segmentation
        net.Process(img)
        net.Mask(class_mask, filter_mode='point')
        mask_array = np.squeeze(jetson_utils.cudaToNumpy(class_mask))

        # Depth
        depth_net.Process(img)
        depth_field = depth_net.GetDepthField()
        depth_array = np.squeeze(jetson_utils.cudaToNumpy(depth_field))

        # Find target pixels
        y_seg, x_seg = np.where(mask_array == target_id)
        pixel_count = len(y_seg)

        if pixel_count > MIN_PIXELS:
            depth_h, depth_w = depth_array.shape
            # Scale mask coords to depth map
            x_depth = np.clip((x_seg * depth_w / grid_w).astype(np.int32), 0, depth_w - 1)
            y_depth = np.clip((y_seg * depth_h / grid_h).astype(np.int32), 0, depth_h - 1)
            latest_depth_val = float(np.mean(depth_array[y_depth, x_depth]))

        # Visualization
        net.Overlay(img, filter_mode='linear')
        
        # Overlay text on the actual video feed
        status_text = f"Class: {target_name} | Pixels: {pixel_count}"
        font.OverlayText(img, img.width, img.height, status_text, 10, 10, (0, 255, 127, 255))
        
        display.Render(img)
        display.SetStatus(f"Grass Tracker | {display.GetFrameRate():.1f} FPS")

# Background thread
thread = threading.Thread(target=camera_loop)
thread.daemon = True
thread.start()

print("\n" + "="*50)
print(f" TRACKING: {target_name.upper()} (ID: {target_id})")
print("="*50 + "\n")

try:
    while running:
        if pixel_count > MIN_PIXELS:
            x = latest_depth_val
            dist_m = max(0, (A * x**3) + (B * x**2) + (C * x) + D)
            print(f"\r[LOCK] Pixels: {pixel_count} | Raw Depth: {x:.4f} | Dist: {dist_m:.3f}m    ", end='', flush=True)
        else:
            print(f"\r[SCANNING...] Pixels: {pixel_count} < {MIN_PIXELS} threshold      ", end='', flush=True)
        time.sleep(0.1)
except KeyboardInterrupt:
    running = False

print("\nShutting down.")
