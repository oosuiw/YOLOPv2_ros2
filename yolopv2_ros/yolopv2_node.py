#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage, PointCloud2, PointField
from std_msgs.msg import Header
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import sys
import os
from pathlib import Path
import time
import struct

# Add YOLOPv2 to path
yolopv2_dir = Path('/home/sws/libraries/YOLOPv2')
sys.path.append(str(yolopv2_dir))

from lib.models import get_net
from lib.config import cfg
from lib.utils.utils import letterbox, show_seg_result, time_synchronized
from utils.utils import LoadImages, non_max_suppression, split_for_trace_model
from lib.utils.plot import plot_one_box


class YOLOPv2Node(Node):
    def __init__(self):
        super().__init__('yolopv2_node')
        
        # Parameters
        self.declare_parameter('weights_path', str(yolopv2_dir / 'data/weights/yolopv2.pt'))
        self.declare_parameter('conf_threshold', 0.3)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('img_size', 640)
        self.declare_parameter('input_topic', '/camera/image_raw')
        self.declare_parameter('output_topic', '/yolopv2/detection_image')
        self.declare_parameter('use_webcam', False)
        self.declare_parameter('webcam_id', 0)
        
        # Get parameters
        self.weights_path = self.get_parameter('weights_path').value
        self.conf_thres = self.get_parameter('conf_threshold').value
        self.iou_thres = self.get_parameter('iou_threshold').value
        self.img_size = self.get_parameter('img_size').value
        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.use_webcam = self.get_parameter('use_webcam').value
        self.webcam_id = self.get_parameter('webcam_id').value
        
        # Initialize model
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.half = self.device.type != 'cpu'
        
        self.get_logger().info(f'Loading YOLOPv2 model from {self.weights_path}')
        self.model = get_net(cfg)
        checkpoint = torch.load(self.weights_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.to(self.device)
        
        if self.half:
            self.model.half()
        
        self.model.eval()
        
        # Warm up
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(next(self.model.parameters())))
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Publishers and subscribers
        if self.use_webcam:
            # Use webcam
            self.cap = cv2.VideoCapture(self.webcam_id)
            if not self.cap.isOpened():
                self.get_logger().error(f'Cannot open webcam {self.webcam_id}')
                return
            
            # Timer for webcam processing
            self.timer = self.create_timer(0.033, self.webcam_callback)  # ~30 FPS
        else:
            # Subscribe to camera topic
            self.subscription = self.create_subscription(
                Image,
                self.input_topic,
                self.image_callback,
                10)
        
        # Publishers
        self.image_pub = self.create_publisher(Image, self.output_topic, 10)
        self.lane_marker_pub = self.create_publisher(MarkerArray, '/yolopv2/lane_markers', 10)
        self.drivable_marker_pub = self.create_publisher(MarkerArray, '/yolopv2/drivable_area', 10)
        self.lane_pointcloud_pub = self.create_publisher(PointCloud2, '/yolopv2/lane_pointcloud', 10)
        self.drivable_pointcloud_pub = self.create_publisher(PointCloud2, '/yolopv2/drivable_pointcloud', 10)
        
        self.get_logger().info('YOLOPv2 Node initialized')
        
    def webcam_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Failed to read from webcam')
            return
        
        self.process_image(frame, is_webcam=True)
    
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.process_image(cv_image)
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')
    
    def process_image(self, img0, is_webcam=False):
        # Resize and preprocess
        img0 = cv2.resize(img0, (1280, 720), interpolation=cv2.INTER_LINEAR)
        img = letterbox(img0, self.img_size, stride=32)[0]
        
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Inference
        t1 = time_synchronized()
        [pred, anchor_grid], seg, ll = self.model(img)
        t2 = time_synchronized()
        
        # Process predictions
        pred = split_for_trace_model(pred, anchor_grid)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None, agnostic=False)
        
        # Process detections
        for i, det in enumerate(pred):
            im0 = img0.copy()
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = self.scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                # Draw boxes
                for *xyxy, conf, cls in reversed(det):
                    plot_one_box(xyxy, im0, line_thickness=3)
            
            # Process segmentation
            da_seg_mask = torch.nn.functional.interpolate(seg, scale_factor=int(1280/self.img_size), mode='bilinear')
            ll_seg_mask = torch.nn.functional.interpolate(ll, scale_factor=int(1280/self.img_size), mode='bilinear')
            
            da_seg_mask = torch.argmax(da_seg_mask, dim=1).squeeze().cpu().numpy()
            ll_seg_mask = torch.argmax(ll_seg_mask, dim=1).squeeze().cpu().numpy()
            
            # Apply segmentation overlay
            show_seg_result(im0, (da_seg_mask, ll_seg_mask), is_demo=True)
            
            if is_webcam:
                # Display for webcam
                cv2.imshow('YOLOPv2 ROS2', im0)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    self.destroy_node()
                    return
            
            # Publish processed image
            try:
                if not is_webcam:
                    ros_image = self.bridge.cv2_to_imgmsg(im0, 'bgr8')
                    ros_image.header.stamp = self.get_clock().now().to_msg()
                    ros_image.header.frame_id = 'camera'
                    self.image_pub.publish(ros_image)
                
                # Publish lane and drivable area markers
                self.publish_markers(ll_seg_mask, da_seg_mask)
                
                # Publish PointCloud2 for lanes and drivable area
                self.publish_pointclouds(ll_seg_mask, da_seg_mask, im0.shape)
                
            except Exception as e:
                self.get_logger().error(f'Error publishing: {str(e)}')
        
        inference_time = t2 - t1
        if not is_webcam:
            self.get_logger().info(f'Inference time: {inference_time:.3f}s')
    
    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        coords[:, :4] = coords[:, :4].clamp(min=0)
        return coords
    
    def publish_markers(self, lane_mask, drivable_mask):
        # Create lane markers
        lane_markers = MarkerArray()
        drivable_markers = MarkerArray()
        
        # Simple marker creation (you can enhance this)
        timestamp = self.get_clock().now().to_msg()
        
        # Lane marker
        lane_marker = Marker()
        lane_marker.header.stamp = timestamp
        lane_marker.header.frame_id = 'camera'
        lane_marker.type = Marker.LINE_STRIP
        lane_marker.action = Marker.ADD
        lane_marker.scale.x = 0.1
        lane_marker.color.r = 1.0
        lane_marker.color.g = 1.0
        lane_marker.color.b = 0.0
        lane_marker.color.a = 1.0
        lane_markers.markers.append(lane_marker)
        
        # Drivable area marker
        drivable_marker = Marker()
        drivable_marker.header.stamp = timestamp
        drivable_marker.header.frame_id = 'camera'
        drivable_marker.type = Marker.TRIANGLE_LIST
        drivable_marker.action = Marker.ADD
        drivable_marker.scale.x = 1.0
        drivable_marker.scale.y = 1.0
        drivable_marker.scale.z = 1.0
        drivable_marker.color.r = 0.0
        drivable_marker.color.g = 1.0
        drivable_marker.color.b = 0.0
        drivable_marker.color.a = 0.3
        drivable_markers.markers.append(drivable_marker)
        
        self.lane_marker_pub.publish(lane_markers)
        self.drivable_marker_pub.publish(drivable_markers)
    
    def publish_pointclouds(self, lane_mask, drivable_mask, img_shape):
        """Convert segmentation masks to PointCloud2 and publish"""
        timestamp = self.get_clock().now().to_msg()
        
        # Create lane pointcloud
        lane_pointcloud = self.mask_to_pointcloud2(lane_mask, timestamp, 'camera', 
                                                  color=[1.0, 0.0, 0.0], mask_value=1)  # Red for lanes
        
        # Create drivable area pointcloud  
        drivable_pointcloud = self.mask_to_pointcloud2(drivable_mask, timestamp, 'camera',
                                                      color=[0.0, 1.0, 0.0], mask_value=1)  # Green for drivable area
        
        # Publish pointclouds
        self.lane_pointcloud_pub.publish(lane_pointcloud)
        self.drivable_pointcloud_pub.publish(drivable_pointcloud)
    
    def mask_to_pointcloud2(self, mask, timestamp, frame_id, color=[1.0, 0.0, 0.0], mask_value=1):
        """Convert segmentation mask to PointCloud2"""
        
        # Find pixels belonging to the segmented class
        y_coords, x_coords = np.where(mask == mask_value)
        
        if len(y_coords) == 0:
            # Return empty pointcloud if no pixels found
            return self.create_empty_pointcloud2(timestamp, frame_id)
        
        # Convert image coordinates to 3D points
        # Assume camera parameters (you can make these configurable)
        fx, fy = 500.0, 500.0  # Focal lengths
        cx, cy = mask.shape[1] / 2, mask.shape[0] / 2  # Principal point
        
        # Assume ground plane at fixed depth for visualization
        z_depth = 5.0  # 5 meters ahead
        
        points = []
        for i in range(len(y_coords)):
            x_img, y_img = x_coords[i], y_coords[i]
            
            # Convert to 3D coordinates (simplified projection)
            x_3d = (x_img - cx) * z_depth / fx
            y_3d = (y_img - cy) * z_depth / fy
            z_3d = z_depth
            
            # Create point with color
            points.append([x_3d, y_3d, z_3d, color[0], color[1], color[2]])
        
        if not points:
            return self.create_empty_pointcloud2(timestamp, frame_id)
        
        # Create PointCloud2 message
        pointcloud = PointCloud2()
        pointcloud.header.stamp = timestamp
        pointcloud.header.frame_id = frame_id
        
        # Define point fields (x, y, z, r, g, b)
        pointcloud.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='r', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='g', offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name='b', offset=20, datatype=PointField.FLOAT32, count=1),
        ]
        
        # Set point cloud properties
        pointcloud.is_bigendian = False
        pointcloud.point_step = 24  # 6 fields * 4 bytes each
        pointcloud.row_step = pointcloud.point_step * len(points)
        pointcloud.is_dense = True
        
        # Pack point data
        pointcloud.data = b''
        for point in points:
            pointcloud.data += struct.pack('ffffff', *point)
        
        pointcloud.width = len(points)
        pointcloud.height = 1
        
        return pointcloud
    
    def create_empty_pointcloud2(self, timestamp, frame_id):
        """Create empty PointCloud2 message"""
        pointcloud = PointCloud2()
        pointcloud.header.stamp = timestamp
        pointcloud.header.frame_id = frame_id
        pointcloud.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='r', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='g', offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name='b', offset=20, datatype=PointField.FLOAT32, count=1),
        ]
        pointcloud.is_bigendian = False
        pointcloud.point_step = 24
        pointcloud.row_step = 0
        pointcloud.width = 0
        pointcloud.height = 1
        pointcloud.is_dense = True
        pointcloud.data = b''
        return pointcloud
    
    def destroy_node(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = YOLOPv2Node()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()