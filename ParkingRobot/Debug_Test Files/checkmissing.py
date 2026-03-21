import os
import subprocess

def check_status(name, command):
    try:
        subprocess.check_output(["which", command])
        print(f"[ OK ] {name}")
        return True
    except:
        print(f"[ MISSING ] {name}")
        return False

print("--- ROS CORE CHECK ---")
ros_core = check_status("ROS Melodic Base", "roscore")
check_status("Catkin Build Tool", "catkin_make")

print("\n--- JETRACER DEPENDENCIES ---")
check_status("RPLidar Driver", "rplidarNode")

# Check for Cartographer
carto = os.path.exists("/opt/ros/melodic/share/cartographer_ros")
print(f"[{' OK ' if carto else ' MISSING '}] Cartographer ROS")

print("\n--- PYTHON 2.7 LIBRARIES (Robot Controls) ---")
try:
    import rospy
    print("[ OK ] rospy (Python 2)")
except ImportError:
    print("[ MISSING ] rospy (Python 2)")

print("\n--- PYTHON 3 LIBRARIES (AI Models) ---")
try:
    import jetson_inference
    import jetson_utils
    print("[ OK ] Jetson Inference (AI)")
except ImportError:
    print("[ MISSING ] Jetson Inference (AI)")

print("\n--- WORKSPACE CHECK ---")
ws_path = os.path.expanduser("~/catkin_ws/src/jetracer_ros")
if os.path.exists(ws_path):
    print(f"[ OK ] JetRacer Source Code found at {ws_path}")
else:
    print(f"[ MISSING ] JetRacer Source Code!")

if not ros_core:
    print("\nHELP: Run 'sudo apt-get install ros-melodic-ros-base' to fix the core.")