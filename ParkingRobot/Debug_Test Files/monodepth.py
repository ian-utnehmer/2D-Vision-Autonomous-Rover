#!/usr/bin/env python3
import jetson_inference
import jetson_utils
import numpy as np

# Basic monodepth test script to check performance on ResNet18.

# Monodepth net, ResNet18 for performance (vs. ResNet50)
net = jetson_inference.depthNet("monodepth-resnet18")

# Setup Camera and WebRTC Output
# View the output at http://<YOUR_IP>:8554
camera = jetson_utils.videoSource("csi://0", argv=['--input-flip=rotate-180'])
display = jetson_utils.videoOutput("webrtc://@:8554/output", argv=['--headless'])

# Pre-allocate depth field buffers
# The raw depth field is a floating-point image, usually 224x224 or similar
depth_field = net.GetDepthField()
depth_numpy = jetson_utils.cudaToNumpy(depth_field)

print("[INFO] Monodepth2 initialized. Streaming to WebRTC...")

# Before the loop, setup a font for the overlay
font = jetson_utils.cudaFont()
try:
    while True:
        img = camera.Capture()
        if img is None: continue

        # Process Depth (This creates the color map)
        net.Process(img)

        # Get the raw data from the center
        y_idx = depth_field.height // 2
        x_idx = depth_field.width // 2
        raw_val = float(depth_numpy[y_idx, x_idx])

        # Draw a Crosshair at the center of the MAIN IMAGE
        # We use img.width/height to scale
        center_img_x = img.width // 2
        center_img_y = img.height // 2

        # Draw a small white circle or crosshair at the measurement point
        jetson_utils.cudaDrawCircle(img, (center_img_x, center_img_y), 5, (255, 255, 255, 255))

        # Overlay the actual value on the video
        font.OverlayText(img, img.width, img.height, f"{raw_val:.2f}",
                         int(center_img_x + 10), int(center_img_y + 10),
                         font.White, font.Gray40)

        # Render to WebRTC
        display.Render(img)


except KeyboardInterrupt:(
    print("\n[INFO] Shutting down..."))