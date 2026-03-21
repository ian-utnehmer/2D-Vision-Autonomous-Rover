import os
import time

# MobaXTerm would occasionally terminate, erasing all active tabs,
# forcing a "reconnect" which left everything still running. In the
# event of MobaXTerm crashing, I made this script to reset all 
# related background processes. 


# pkill commands
os.system("sudo pkill -f detectnet")
os.system("sudo pkill -f video-viewer")
os.system("sudo pkill -f gst-launch-1.0")
os.system("sudo pkill -f python")

# Restart the camera daemon
os.system("sudo systemctl restart nvargus-daemon")

# Wait for hardware initialization
time.sleep(2)