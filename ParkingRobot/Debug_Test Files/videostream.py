#!/usr/bin/env python3
import time
import jetson_utils

# Test camera script
camera = jetson_utils.videoSource("csi://0", argv=[
    '--input-width=1920',
    '--input-height=1080',
    '--input-rate=30'
])

# WebRTC, e.g., can further use VLC to view stream
display = jetson_utils.videoOutput("webrtc://<YOUR_IP>:8554")

print("warming up...")
time.sleep(2.0)
print("Open <YOUR_IP>:8554")

while True:
    img = camera.Capture(timeout=2000)
    if img is None:
        continue

    display.Render(img)
    display.SetStatus("camera test")

    if not camera.IsStreaming():
        break