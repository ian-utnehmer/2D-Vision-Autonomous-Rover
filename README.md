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
roslaunch jetracer jetracer_paper.launch
roslaunch jetracer move_base.launch
python3 /publishing/pointcloud.py
```

Then, view in RViz.






### ROS Launch & Configuration
* **`jetracer_paper.launch`**: A hardware-level launch file that sets the specific polynomial coefficients.
* **`jetracer_paper.lua`**: The configuration script for **Google Cartographer**, defining tracking frames, distances, and sampling rates for 2D SLAM.
* **`jetracer.urdf`**: The Unified Robot Description Format file. It defines the physical dimensions and visual properties of the JetRacer, including the positions of the camera and LiDAR relative to the base.

### Navigation Stack Parameters
* **`costmap_common_params.yaml`**: Defines settings shared by both local and global costmaps, such as the robot’s footprint and the standard LiDAR obstacle layer.
* **`global_costmap_params.yaml`**: Configures the global map (used for long-term planning).
* **`local_costmap_params.yaml`**: Configures the rolling window around the robot. Ensures the AI detections (paper/grass) are cleared or marked in real-time as the robot moves.
* **`move_base_params.yaml`**: Selects the navigation planners (GlobalPlanner and TEB Local Planner) and defines recovery behaviors.

### Perception & AI Inference
* **`pointcloud.py`**: The main python script. Converts the 2D semantic segmentation masks (from Cityscapes) into a 3D `PointCloud2` message by projecting pixels into 3D space using Monodepth2 data.
* **`cityscapes.py`**: A diagnostic script that runs the full Cityscapes model to visualize all 19-21 detectable classes (roads, people, vegetation, etc.) via WebRTC.
* **`grassview.py` / `grassview2.py`**: These scripts focus specifically on the "Vegetation" class. They track the density of grass in the frame and calculate its distance from the robot using the calibration coefficients.
* **`monodepth.py`**: A test script for the Monodepth2 model to verify depth estimation performance and frame rates on the Jetson Nano.

### Training & Calibration
* **`image_sampling.py`**: A utility used during the data collection phase to capture and save images from the CSI camera to build a custom training dataset.
* **`train_resnet18_fpn.py`**: The PyTorch training script. It trains the segmentation model on a PC and exports the final result as a Jetson-optimized `.onnx` file.
* **`paper_calibration.py`**: A manual calibration tool. It records raw AI depth values alongside physical measurements (inches) provided by the user to create a mapping dataset (`paper_data_inches.csv`).
* **`training_script.py`**: Processes the calibration data using `numpy.polyfit` to generate the 3rd-degree polynomial coefficients used in the ROS nodes.
* **`paper_test_coefficients.py`**: A live validation script that uses the generated $A, B, C, D$ coefficients to show real-time distance estimates for accuracy checking.
* **`paper_depth_node.py`**: A ROS node used to monitor and print live distance calculations (raw vs. meters) during testing.

### Utilities & Diagnostics
* **`checkmissing.py`**: A system health script that verifies all necessary ROS packages, Python libraries, and workspace paths are correctly configured.
* **`videostream.py`**: A simple test script to verify the CSI camera feed and WebRTC streaming functionality independently of the AI models.
* **`restart.py`**: A convenience script that kills hanging camera processes and restarts the `nvargus-daemon` to clear hardware errors.



