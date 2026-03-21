# JetRacer Autonomous Navigation: Dual-Vision Obstacle Avoidance

This project implements an autonomous navigation stack for a **Waveshare JetRacer** (NVIDIA Jetson Nano) that overcomes the limitations of 2D LiDAR by integrating real-time semantic segmentation and monocular depth estimation.

## Project Overview
Standard 2D LiDAR sensors (like the RPLIDAR A1) often fail to detect low-profile obstacles or differentiate between floor textures. This system utilizes a CSI camera to identify "non-drivable" surfaces—specifically **paper** and **grass**—and projects these detections into the ROS navigation costmap as virtual obstacles.

### Key Objectives:
* **LiDAR + Vision Fusion:** Combining 2D SLAM with AI-driven terrain analysis.
* **Terrain Awareness:** Avoiding grass/vegetation using the Cityscapes model.
* **Custom Object Avoidance:** Detecting and avoiding paper obstacles via a custom-trained ResNet18-FPN model.
* **Precise Navigation:** Utilizing the TEB Local Planner for dynamic path planning around both physical and "visual" obstacles.

## System Architecture
The system operates on **ROS Melodic** and consists of several interconnected modules:

* **SLAM & Localization:** Google Cartographer handles 2D mapping and state estimation.
* **Perception:**
    * **ResNet18-FPN:** Custom trained to detect paper.
    * **Cityscapes (fcn-resnet18):** Pre-trained model used to segment "nature/vegetation" classes.
    * **Monodepth2:** Provides distance estimation from a single RGB feed.
* **Navigation:** `move_base` with a custom `paper_layer` (ObstacleLayer) that receives `PointCloud2` data generated from the camera feed.

## Hardware Requirements
* **Platform:** Waveshare JetRacer (AI Kit).
* **Compute:** NVIDIA Jetson Nano.
* **Sensors:** * RPLIDAR A1 (2D LiDAR).
    * IMX219-160 8MP CSI Camera.
* **Connectivity:** MobaXTerm/NoMachine for remote management; WebRTC for low-latency video streaming.

## Usage
### Perception Calibration
Before navigating, the depth-to-distance mapping must be calibrated using a ruler:
1.  Run `paper_calibration.py` to collect raw AI depth values vs. physical inches.
2.  Run `training_script.py` to generate the 3rd-degree polynomial coefficients (A, B, C, D).
3.  Update the coefficients in your launch files or `grassview.py`.

### Launching Navigation
To start the full SLAM and navigation stack with paper/grass avoidance:
```bash
roslaunch jetracer slam_paper.launch
