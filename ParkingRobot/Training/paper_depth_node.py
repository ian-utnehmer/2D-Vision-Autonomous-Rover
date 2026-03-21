#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32

last_raw = 0.0

def raw_cb(msg):
    global last_raw
    last_raw = msg.data

def dist_cb(msg):
    # This prints whenever a new distance is calculated
    print(f"RAW AI: {last_raw:.4f}  |  CALCULATED: {msg.data:.3f}m  |  (~{msg.data*39.37:.1f} inches)")

rospy.init_node('calibration_tester')
rospy.Subscriber('/paper_project/raw_depth', Float32, raw_cb)
rospy.Subscriber('/paper_project/distance_meters', Float32, dist_cb)

print("--- STARTING LIVE TEST ---")
rospy.spin()