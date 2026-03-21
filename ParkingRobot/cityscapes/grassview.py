#!/usr/bin/env python3
import jetson_inference
import jetson_utils
import numpy as np
import threading
import time
import sys

# --- CONFIG ---
MODEL_NAME = "fcn-resnet18-cityscapes"
VEGETATION_ID = 21  # Cityscapes 'nature' class
MIN_PIXELS = 250

# --- CALIBRATION ---
A, B, C, D = 0.3045211441, -2.2437643455, 5.6336667304, -4.3246513581

# --- GLOBALS ---
latest_depth_val = 0.0
pixel_count = 0
running = True

# --- INIT ---
# Load Networks
net = jetson_inference.segNet(MODEL_NAME)
depth_net = jetson_inference.depthNet(argv=['--model=monodepth-resnet18'])

# Set Overlay (150 Alpha = Camera + Labels)
net.SetOverlayAlpha(150.0)

# Setup IO
# Input from CSI Camera
camera = jetson_utils.videoSource("csi://0", argv=[
    '--input-width=1280', '--input-height=720', '--input-rate=15', '--input-flip=rotate-180'
])

# Output to NATIVE WINDOW (Works via NoMachine)
display = jetson_utils.videoOutput("display://0")

# Prepare Mask Buffer
grid_w, grid_h = net.GetGridWidth(), net.GetGridHeight()
class_mask = jetson_utils.cudaAllocMapped(width=grid_w, height=grid_h, format='gray8')

def camera_loop():
    global latest_depth_val, pixel_count, running
    while running:
        img = camera.Capture()
        if img is None: continue

        # --- SEGMENTATION ---
        net.Process(img)
        net.Mask(class_mask, filter_mode='point')
        mask_array = np.squeeze(jetson_utils.cudaToNumpy(class_mask))

        # --- DEPTH ---
        depth_net.Process(img)
        depth_field = depth_net.GetDepthField()
        depth_array = np.squeeze(jetson_utils.cudaToNumpy(depth_field))

        # --- MATH ---
        y_seg, x_seg = np.where(mask_array == VEGETATION_ID)
        pixel_count = len(y_seg)

        if pixel_count > MIN_PIXELS:
            depth_h, depth_w = depth_array.shape
            # Map SegNet grid coords to DepthNet field coords
            x_depth = np.clip((x_seg * depth_w / grid_w).astype(np.int32), 0, depth_w - 1)
            y_depth = np.clip((y_seg * depth_h / grid_h).astype(np.int32), 0, depth_h - 1)
            latest_depth_val = float(np.mean(depth_array[y_depth, x_depth]))

        # --- VISUALIZATION ---
        # Paint the labels onto the camera frame
        net.Overlay(img, filter_mode='linear')
        
        # Render the frame to the Desktop Window
        display.Render(img)
        
        # Update window title with FPS
        display.SetStatus(f"Cityscapes Grass Tracker | {display.GetFrameRate():.1f} FPS")

# Start Thread
thread = threading.Thread(target=camera_loop)
thread.daemon = True
thread.start()

print("\n" + "="*50)
print("  NATIVE DESKTOP TRACKER ACTIVE")
print("  Check your NoMachine window for the video feed.")
print("="*50 + "\n")

try:
    while running:
        if pixel_count > MIN_PIXELS:
            x = latest_depth_val
            dist_m = max(0, (A * x**3) + (B * x**2) + (C * x) + D)
            print(f"\r[LOCK] Veg Pixels: {pixel_count} | Dist: {dist_m:.3f}m", end='', flush=True)
        else:
            print(f"\r[SCANNING...] Looking for vegetation...          ", end='', flush=True)
        time.sleep(0.1)
except KeyboardInterrupt:
    running = False

print("\nExiting...")
