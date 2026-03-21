#!/usr/bin/env python3
import jetson_inference
import jetson_utils
import numpy as np
import csv
import os
import threading
import sys
import select
import time

# CONFIG
SEG_MODEL = "/home/jetson/paper_project/models/paper_resnet18_fpn/paper_resnet18_unet.onnx"
LABELS = "/home/jetson/paper_project/models/paper_resnet18_fpn/labels.txt"
DATA_FILE = "paper_data_inches.csv"
PAPER_CLASS_ID = 1
MIN_PIXELS = 250

# GLOBALS
latest_depth_val = 0.0
pixel_count = 0
running = True

# INIT
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
        # Same process code as before
        img = camera.Capture()
        if img is None: continue
        net.Process(img)
        net.Mask(class_mask, filter_mode='point')
        
        # Squeeze to flatten
        mask_array = np.squeeze(jetson_utils.cudaToNumpy(class_mask))
        depth_net.Process(img)
        depth_field = depth_net.GetDepthField()
        depth_array = np.squeeze(jetson_utils.cudaToNumpy(depth_field))
        y_seg, x_seg = np.where(mask_array == PAPER_CLASS_ID)
        pixel_count = len(y_seg)
        if pixel_count > MIN_PIXELS:
            # MIN_PIXELS to avoid reads due to light refraction
            depth_h, depth_w = depth_array.shape
            x_depth = np.clip((x_seg * depth_w / seg_w).astype(np.int32), 0, depth_w - 1)
            y_depth = np.clip((y_seg * depth_h / seg_h).astype(np.int32), 0, depth_h - 1)
            latest_depth_val = float(np.mean(depth_array[y_depth, x_depth]))
        net.Overlay(img, filter_mode='linear')
        display.Render(img)


def get_input_non_blocking():
    if select.select([sys.stdin], [], [], 0.1)[0]:
        return sys.stdin.readline()
    return None


# Threading due to camera framerate issues
thread = threading.Thread(target=camera_loop)
thread.daemon = True
thread.start()

print("\n" + "=" * 50)
print("  STEP 1: PLACE PAPER | STEP 2: HIT ENTER | STEP 3: TYPE INCHES")
print("=" * 50)

try:
    while running:
        if pixel_count > MIN_PIXELS:
            print(f"\r[LOCKED] Raw: {latest_depth_val:.4f} | Pixels: {pixel_count} -- PRESS ENTER TO LOG --", end='',
                  flush=True)

            if get_input_non_blocking() is not None:
                print("\n\n" + "!" * 40)
                val = input(" >>> TYPE THE DISTANCE IN INCHES: ")
                print("!" * 40 + "\n")

                if val.lower() == 'q':
                    running = False
                    break
                try:
                    dist_inches = float(val)
                    file_exists = os.path.isfile(DATA_FILE)
                    with open(DATA_FILE, 'a', newline='') as f:
                        writer = csv.writer(f)
                        if not file_exists: writer.writerow(["raw_depth", "inches"])
                        writer.writerow([latest_depth_val, dist_inches])
                    print(f"RECORDED: {latest_depth_val:.4f} @ {dist_inches}in")
                except ValueError:
                    print("Error: NaN.")
        else:
            print(f"\r[SEARCHING...] Pixels: {pixel_count}          ", end='', flush=True)
            time.sleep(0.1)

except KeyboardInterrupt:
    running = False

print("\nDone. CSV saved.")