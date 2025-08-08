#!/usr/bin/env python3

import sys
import os
sys.path.append('/home/sws/libraries/YOLOPv2')

# 기존 demo.py를 ROS2 래퍼로 실행
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import time
from pathlib import Path
import struct

# YOLOPv2 imports
import demo

class YOLOPv2ROSWrapper(Node):
    def __init__(self):
        super().__init__('yolopv2_ros_wrapper')
        
        # Parameters
        self.declare_parameter('use_webcam', True)
        self.declare_parameter('webcam_id', 0)
        self.declare_parameter('input_topic', '/camera/image_raw')
        
        self.use_webcam = self.get_parameter('use_webcam').value
        self.webcam_id = self.get_parameter('webcam_id').value
        self.input_topic = self.get_parameter('input_topic').value
        
        # CV bridge
        self.bridge = CvBridge()
        
        # Publishers
        self.image_pub = self.create_publisher(Image, '/yolopv2/detection_image', 10)
        self.lane_pointcloud_pub = self.create_publisher(PointCloud2, '/yolopv2/lane_pointcloud', 10)
        self.drivable_pointcloud_pub = self.create_publisher(PointCloud2, '/yolopv2/drivable_pointcloud', 10)
        
        if self.use_webcam:
            self.cap = cv2.VideoCapture(self.webcam_id)
            if not self.cap.isOpened():
                self.get_logger().error(f'Cannot open webcam {self.webcam_id}')
                return
            self.timer = self.create_timer(0.033, self.webcam_callback)
        else:
            self.subscription = self.create_subscription(
                Image, self.input_topic, self.image_callback, 10)
        
        self.get_logger().info('YOLOPv2 ROS Wrapper initialized')
    
    def webcam_callback(self):
        ret, frame = self.cap.read()
        if ret:
            self.process_frame(frame)
    
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.process_frame(cv_image)
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')
    
    def process_frame(self, frame):
        # 간단한 세그멘테이션 시뮬레이션 (실제로는 YOLOPv2 모델 사용)
        # 실제 구현을 위해서는 YOLOPv2의 inference 로직을 여기에 통합해야 합니다
        
        # 임시로 화면 하단을 차선으로 가정
        h, w = frame.shape[:2]
        lane_mask = np.zeros((h, w), dtype=np.uint8)
        drivable_mask = np.zeros((h, w), dtype=np.uint8)
        
        # 하단 중앙부를 차선으로 표시
        lane_mask[int(h*0.7):h, int(w*0.3):int(w*0.7)] = 1
        drivable_mask[int(h*0.5):h, int(w*0.2):int(w*0.8)] = 1
        
        # 결과 이미지 생성
        result_img = frame.copy()
        result_img[lane_mask == 1] = [0, 0, 255]  # 빨간색 차선
        result_img = cv2.addWeighted(frame, 0.7, result_img, 0.3, 0)
        
        # 웹캠인 경우 화면 표시
        if self.use_webcam:
            cv2.imshow('YOLOPv2 ROS2', result_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                self.destroy_node()
                return
        
        # ROS 이미지 퍼블리시
        try:
            ros_image = self.bridge.cv2_to_imgmsg(result_img, 'bgr8')
            ros_image.header.stamp = self.get_clock().now().to_msg()
            ros_image.header.frame_id = 'camera'
            self.image_pub.publish(ros_image)
            
            # PointCloud2 퍼블리시
            self.publish_pointclouds(lane_mask, drivable_mask)
            
        except Exception as e:
            self.get_logger().error(f'Error publishing: {str(e)}')
    
    def publish_pointclouds(self, lane_mask, drivable_mask):
        timestamp = self.get_clock().now().to_msg()
        
        # Lane pointcloud (빨간색)
        lane_pc = self.mask_to_pointcloud2(lane_mask, timestamp, 'camera', [1.0, 0.0, 0.0])
        self.lane_pointcloud_pub.publish(lane_pc)
        
        # Drivable area pointcloud (초록색)  
        drivable_pc = self.mask_to_pointcloud2(drivable_mask, timestamp, 'camera', [0.0, 1.0, 0.0])
        self.drivable_pointcloud_pub.publish(drivable_pc)
    
    def mask_to_pointcloud2(self, mask, timestamp, frame_id, color):
        y_coords, x_coords = np.where(mask == 1)
        
        if len(y_coords) == 0:
            return self.create_empty_pointcloud2(timestamp, frame_id)
        
        # 카메라 파라미터 (가정)
        fx, fy = 500.0, 500.0
        cx, cy = mask.shape[1] / 2, mask.shape[0] / 2
        z_depth = 5.0
        
        points = []
        # 포인트 수를 줄여서 성능 향상
        step = max(1, len(y_coords) // 1000)  # 최대 1000개 포인트
        
        for i in range(0, len(y_coords), step):
            x_img, y_img = x_coords[i], y_coords[i]
            x_3d = (x_img - cx) * z_depth / fx
            y_3d = (y_img - cy) * z_depth / fy
            points.append([x_3d, y_3d, z_depth, color[0], color[1], color[2]])
        
        # PointCloud2 생성
        pc = PointCloud2()
        pc.header.stamp = timestamp
        pc.header.frame_id = frame_id
        pc.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='r', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='g', offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name='b', offset=20, datatype=PointField.FLOAT32, count=1),
        ]
        pc.is_bigendian = False
        pc.point_step = 24
        pc.row_step = pc.point_step * len(points)
        pc.is_dense = True
        pc.width = len(points)
        pc.height = 1
        
        pc.data = bytearray()
        for point in points:
            pc.data.extend(struct.pack('ffffff', *point))
        
        return pc
    
    def create_empty_pointcloud2(self, timestamp, frame_id):
        pc = PointCloud2()
        pc.header.stamp = timestamp
        pc.header.frame_id = frame_id
        pc.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='r', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='g', offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name='b', offset=20, datatype=PointField.FLOAT32, count=1),
        ]
        pc.is_bigendian = False
        pc.point_step = 24
        pc.width = 0
        pc.height = 1
        pc.row_step = 0
        pc.is_dense = True
        pc.data = bytearray()
        return pc
    
    def destroy_node(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    try:
        node = YOLOPv2ROSWrapper()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()