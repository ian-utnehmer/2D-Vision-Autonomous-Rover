#!/usr/bin/env python3
import sys
import os

# Ensure paths for Jetson Melodic
sys.path.append('/usr/lib/python3.6/dist-packages')
sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages')

import rospy
import numpy as np
import struct
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

try:
    import jetson_inference
    import jetson_utils
except ImportError as e:
    print(f"CRITICAL: {e}")
    sys.exit(1)

class CityscapesGrassAvoidance:
    def __init__(self):
        rospy.init_node('cityscapes_grass_node', anonymous=True)
        
        self.pub = rospy.Publisher('/paper_project/cloud', PointCloud2, queue_size=1)

        # Load Cityscapes and Monodepth
        self.net = jetson_inference.segNet("fcn-resnet18-cityscapes")
        self.depth_net = jetson_inference.depthNet("monodepth-resnet18")

        # Camera settings from paper script
        self.camera = jetson_utils.videoSource("csi://0", argv=[
            '--input-width=640', '--input-height=360', '--input-flip=rotate-180'
        ])

        # Grid setup
        self.mask_width = self.net.GetGridWidth()
        self.mask_height = self.net.GetGridHeight()
        self.class_mask = jetson_utils.cudaAllocMapped(width=self.mask_width, height=self.mask_height, format='gray8')

        # DYNAMIC ID DETECTION 
        self.target_id = 21 # Default for Cityscapes, check below (as we previously briefly tested with other models)
        for n in range(self.net.GetNumClasses()):
            if 'vegetation' in self.net.GetClassDesc(n).lower():
                self.target_id = n
                break
        
        rospy.loginfo(f"Grass Node Online. Targeting Class ID: {self.target_id}")

    def create_pc2(self, points):
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "base_footprint"

        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1),
        ]

        point_data = [struct.pack('ffff', float(p[0]), float(p[1]), float(p[2]), 255.0) for p in points]
        
        # Pointcloud of scanned data
        return PointCloud2(
            header=header, height=1, width=len(points), is_dense=False,
            is_bigendian=False, fields=fields,
            point_step=16, row_step=16 * len(points),
            data=b''.join(point_data)
        )

    def run(self):
        # Actual camera constants
        focal_length = 185.0
        depth_scale = 0.3
        camera_offset_x = 0.15
        center_x = self.mask_width / 2.0

        while not rospy.is_shutdown():
            img = self.camera.Capture()
            if img is None: continue

            # Run Inference
            self.net.Process(img)
            self.depth_net.Process(img)
            self.net.Mask(self.class_mask, self.mask_width, self.mask_height, filter_mode='point')

            mask_array = np.squeeze(jetson_utils.cudaToNumpy(self.class_mask))
            depth_field = self.depth_net.GetDepthField()
            depth_array = np.squeeze(jetson_utils.cudaToNumpy(depth_field))

            # Find Vegetation (Grass)
            y_idx, x_idx = np.where(mask_array == self.target_id)

            if len(x_idx) > 50:
                points = []
                # Downsample by 5 (tuned constant)
                for i in range(0, len(x_idx), 5):
                    u, v = x_idx[i], y_idx[i]
                    
                    # Coordinate scaling
                    d_u = int(u * depth_array.shape[1] / self.mask_width)
                    d_v = int(v * depth_array.shape[0] / self.mask_height)

                    # Projection logic
                    z_m = depth_array[d_v, d_u] * depth_scale
                    x_m = (u - center_x) * z_m / focal_length

                    ros_x = z_m + camera_offset_x
                    ros_y = -x_m
                    ros_z = 0.05  # Height for costmap hit

                    points.append([ros_x, ros_y, ros_z])

                # Publish for viewing in RViz
                self.pub.publish(self.create_pc2(points))
                
            else:
                # Send empty data so jetracer_launch doesn't hang
                # Send empty data so RViz doesn't hang
                self.pub.publish(self.create_pc2([]))

if __name__ == '__main__':
    try:
        node = CityscapesGrassAvoidance()
        node.run()
    except rospy.ROSInterruptException:
        pass
