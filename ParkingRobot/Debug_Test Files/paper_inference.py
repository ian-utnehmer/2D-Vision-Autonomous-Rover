#!/usr/bin/env python3
import time
import numpy as np
import jetson_inference
import jetson_utils
import rospy
from sensor_msgs.msg import LaserScan


# ROS Init
rospy.init_node('paper_parking_scout', anonymous=True)
scan_pub = rospy.Publisher('/paper_scan', LaserScan, queue_size=10)

# Config & Models
PAPER_CLASS_ID = 1
MIN_PAPER_PIXELS = 500
DANGER_DISPARITY = 1.3

# Trained .onnx file (output of training script)
SEG_MODEL = "/home/jetson/paper_project/models/paper_resnet18_fpn/paper_resnet18_unet.onnx"
LABELS = "/home/jetson/paper_project/models/paper_resnet18_fpn/labels.txt"

# SegNet + Monodepth launch
net = jetson_inference.segNet(
    argv=[f'--model={SEG_MODEL}', f'--labels={LABELS}', '--input_blob=input_0', '--output_blob=output_0'])
depth_net = jetson_inference.depthNet(argv=['--model=monodepth-resnet18'])

# Viewstream, again on IP:8554
camera = jetson_utils.videoSource("csi://0", argv=['--input-width=640', '--input-height=360', '--input-rate=30',
                                                   '--input-flip=rotate-180'])
display = jetson_utils.videoOutput("webrtc://@:8554/output", argv=['--headless'])

seg_w, seg_h = net.GetGridWidth(), net.GetGridHeight()
class_mask = jetson_utils.cudaAllocMapped(width=seg_w, height=seg_h, format='gray8')

# Main Loop
while not rospy.is_shutdown():
    t0 = time.time()
    img = camera.Capture(timeout=1000)
    if img is None: continue

    # Segmentation
    net.Process(img)
    net.Mask(class_mask, filter_mode='point')
    mask_array = np.squeeze(jetson_utils.cudaToNumpy(class_mask))

    # Depth
    depth_net.Process(img)
    jetson_utils.cudaDeviceSynchronize()
    depth_field = depth_net.GetDepthField()
    depth_array = np.squeeze(jetson_utils.cudaToNumpy(depth_field))
    depth_h, depth_w = depth_array.shape

    # Process Detections
    y_seg, x_seg = np.where(mask_array == PAPER_CLASS_ID)
    num_pixels = len(y_seg)

    if num_pixels >= MIN_PAPER_PIXELS:
        # Map Coordinates
        x_depth = np.clip((x_seg * depth_w / seg_w).astype(np.int32), 0, depth_w - 1)
        y_depth = np.clip((y_seg * depth_h / seg_h).astype(np.int32), 0, depth_h - 1)

        depth_val = float(np.mean(depth_array[y_depth, x_depth]))

        # Calculate Distance (Inverse of Disparity)
        # The 2.0 constant is tuned similarly to tuning KD, KI, KP values
        # Adjust until accuracy fits real-world testing w/ ruler and distances.
        dist_m = 2.0 / (depth_val + 0.01)

        # Calculate Angle (Center of paper relative to image center)
        avg_x = np.mean(x_seg)
        
        # Camera FOV is roughly 60 degrees for this test 
        # However, the camera's actual FoV is 120 deg.
        
        angle_rad = ((avg_x / seg_w) - 0.5) * 1.04

        # Build LaserScan Message
        scan = LaserScan()
        scan.header.stamp = rospy.Time.now()
        scan.header.frame_id = "camera_link"  # Must exist in TF tree
        scan.angle_min = angle_rad - 0.01
        scan.angle_max = angle_rad + 0.01
        scan.angle_increment = 0.01
        scan.range_min = 0.1
        scan.range_max = 10.0
        scan.ranges = [dist_m]

        scan_pub.publish(scan)

        status = "TOO CLOSE" if depth_val >= DANGER_DISPARITY else "SAFE"
        print(f"{status} | depth={depth_val:.2f} | dist={dist_m:.2f}m | pixels={num_pixels}", flush=True)
    else:
        print("Scouting...", end='\r', flush=True)

    net.Overlay(img, filter_mode='linear')
    display.Render(img)