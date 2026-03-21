#!/usr/bin/env python3
import os
import time
import cv2
import jetson_utils

# Create a folder to save images
SAVE_DIR = "dataset"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

camera = jetson_utils.videoSource("csi://0", argv=['--input-width=640', '--input-height=480'])

print("Capturing images... Move the paper around! Press Ctrl+C to stop.")
img_count = 0

try:
    while True:
        img = camera.Capture()
        if img is None: continue

        # Save one frame every 2 seconds
        if img_count % 60 == 0:
            timestamp = int(time.time())
            # Convert to OpenCV to save
            cuda_mem = jetson_utils.cudaToNumpy(img)
            bgr = cv2.cvtColor(cuda_mem, cv2.COLOR_RGBA2BGR)
            bgr = cv2.rotate(bgr, cv2.ROTATE_180)

            filename = os.path.join(SAVE_DIR, f"paper_{timestamp}.jpg")
            cv2.imwrite(filename, bgr)
            print(f"Saved: {filename}")

        img_count += 1
        time.sleep(0.01)
except KeyboardInterrupt:
    print("\nCapture stopped. Check the 'dataset' folder!")