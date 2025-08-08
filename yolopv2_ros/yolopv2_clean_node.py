#!/usr/bin/env python3
"""
Clean and optimized YOLOPv2 ROS2 node
- Consolidated functionality
- Removed code duplication
- Simplified configuration
- Better error handling
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import time
import struct
import os
import sys
from pathlib import Path

# Add YOLOPv2 path dynamically
def find_yolopv2_path():
    """Find YOLOPv2 installation path"""
    possible_paths = [
        '/home/sws/libraries/YOLOPv2',
        os.path.expanduser('~/YOLOPv2'),
        os.path.expanduser('~/libraries/YOLOPv2'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    raise FileNotFoundError("YOLOPv2 installation not found. Please check installation path.")

try:
    YOLOPV2_PATH = find_yolopv2_path()
    sys.path.append(YOLOPV2_PATH)
    
    # Import YOLOPv2 utilities
    from utils.utils import letterbox, show_seg_result, time_synchronized, non_max_suppression, split_for_trace_model
    from lib.utils.plot import plot_one_box
    
except Exception as e:
    print(f"Warning: Could not import YOLOPv2 modules: {e}")
    YOLOPV2_PATH = None


class YOLOPv2CleanNode(Node):
    """Clean, consolidated YOLOPv2 ROS2 node"""
    
    def __init__(self):
        super().__init__('yolopv2_clean_node')
        
        # Declare essential parameters only
        self.declare_parameters()
        self.load_parameters()
        
        # Initialize components
        self.bridge = CvBridge()
        self.model = None
        self.device = None
        
        # Setup model
        if not self.initialize_model():
            self.get_logger().error("Failed to initialize model. Using dummy mode.")
        
        # Setup publishers
        self.setup_publishers()
        
        # Setup input (webcam or ROS topic)
        self.setup_input()
        
        self.get_logger().info(f"YOLOPv2 Clean Node initialized")
        self.get_logger().info(f"  Mode: {'Webcam' if self.use_webcam else 'ROS Topic'}")
        self.get_logger().info(f"  Device: {self.device}")
        self.get_logger().info(f"  Model: {'Loaded' if self.model else 'Dummy Mode'}")
    
    def declare_parameters(self):
        """Declare only essential parameters"""
        # Model parameters
        self.declare_parameter('model_path', 'model/yolopv2.pt')
        self.declare_parameter('use_gpu', True)
        self.declare_parameter('use_half_precision', True)
        
        # Input/Output parameters
        self.declare_parameter('use_webcam', False)
        self.declare_parameter('webcam_id', 0)
        self.declare_parameter('input_topic', '/camera/image_raw')
        self.declare_parameter('output_topic', '/yolopv2/detection_image')
        
        # Processing parameters
        self.declare_parameter('conf_threshold', 0.3)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('img_size', 640)
        self.declare_parameter('show_window', True)
        
        # PointCloud parameters
        self.declare_parameter('publish_pointclouds', True)
        self.declare_parameter('max_points', 1000)
        self.declare_parameter('depth', 5.0)
    
    def load_parameters(self):
        """Load parameters with validation"""
        # Model parameters
        self.model_path = self.get_parameter('model_path').value
        self.use_gpu = self.get_parameter('use_gpu').value
        self.use_half_precision = self.get_parameter('use_half_precision').value
        
        # Input/Output
        self.use_webcam = self.get_parameter('use_webcam').value
        self.webcam_id = self.get_parameter('webcam_id').value
        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        
        # Processing
        self.conf_threshold = self.get_parameter('conf_threshold').value
        self.iou_threshold = self.get_parameter('iou_threshold').value
        self.img_size = self.get_parameter('img_size').value
        self.show_window = self.get_parameter('show_window').value
        
        # PointCloud
        self.publish_pointclouds = self.get_parameter('publish_pointclouds').value
        self.max_points = self.get_parameter('max_points').value
        self.depth = self.get_parameter('depth').value
        
        # Resolve model path
        if not os.path.isabs(self.model_path):
            # Try package model directory first
            package_model = os.path.join(os.path.dirname(__file__), '..', self.model_path)
            if os.path.exists(package_model):
                self.model_path = package_model
            else:
                self.get_logger().warn(f"Model not found at {package_model}")
    
    def initialize_model(self):
        """Initialize YOLOPv2 model"""
        try:
            # Setup device
            if self.use_gpu and torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
                self.use_half_precision = False  # No half precision on CPU
            
            # Check model file
            if not os.path.exists(self.model_path):
                self.get_logger().error(f"Model file not found: {self.model_path}")
                return False
            
            self.get_logger().info(f"Loading model from: {self.model_path}")
            
            # Load TorchScript model
            self.model = torch.jit.load(self.model_path, map_location=self.device)
            
            if self.use_half_precision:
                self.model = self.model.half()
            
            self.model.eval()
            
            # Warm up
            self.warmup_model()
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"Model initialization failed: {e}")
            return False
    
    def warmup_model(self):
        """Warm up model with dummy input"""
        if self.device.type != 'cpu' and self.model:
            try:
                dummy_input = torch.zeros(1, 3, self.img_size, self.img_size).to(self.device)
                if self.use_half_precision:
                    dummy_input = dummy_input.half()
                
                with torch.no_grad():
                    _ = self.model(dummy_input)
                
                del dummy_input
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                self.get_logger().info("Model warmed up successfully")
                
            except Exception as e:
                self.get_logger().warn(f"Model warmup failed: {e}")
    
    def setup_publishers(self):
        """Setup ROS publishers"""
        self.image_pub = self.create_publisher(Image, self.output_topic, 10)
        
        if self.publish_pointclouds:
            self.lane_pc_pub = self.create_publisher(
                PointCloud2, '/yolopv2/lane_pointcloud', 10)
            self.drivable_pc_pub = self.create_publisher(
                PointCloud2, '/yolopv2/drivable_pointcloud', 10)
    
    def setup_input(self):
        """Setup input source (webcam or ROS topic)"""
        if self.use_webcam:
            self.setup_webcam()
        else:
            self.setup_ros_input()
    
    def setup_webcam(self):
        """Setup webcam input"""
        try:
            self.cap = cv2.VideoCapture(self.webcam_id)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open webcam {self.webcam_id}")
            
            # Set webcam properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Timer for webcam processing (~30 FPS)
            self.timer = self.create_timer(0.033, self.webcam_callback)
            
            self.get_logger().info(f"Webcam {self.webcam_id} initialized")
            
        except Exception as e:
            self.get_logger().error(f"Webcam setup failed: {e}")
            self.use_webcam = False
            self.setup_ros_input()
    
    def setup_ros_input(self):
        """Setup ROS topic input"""
        self.subscription = self.create_subscription(
            Image, self.input_topic, self.image_callback, 10)
        self.get_logger().info(f"Subscribed to: {self.input_topic}")
    
    def webcam_callback(self):
        """Webcam timer callback"""
        if hasattr(self, 'cap'):
            ret, frame = self.cap.read()
            if ret:
                self.process_frame(frame, from_webcam=True)
            else:
                self.get_logger().warn("Failed to read from webcam")
    
    def image_callback(self, msg):
        """ROS image callback"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.process_frame(cv_image, from_webcam=False)
        except Exception as e:
            self.get_logger().error(f"Image callback error: {e}")
    
    def process_frame(self, frame, from_webcam=False):
        """Main frame processing function"""
        try:
            if self.model and YOLOPV2_PATH:
                result_img, lane_mask, drivable_mask, inference_time = self.yolopv2_inference(frame)
                if not from_webcam:
                    self.get_logger().info(f"Inference: {inference_time:.3f}s")
            else:
                result_img, lane_mask, drivable_mask = self.dummy_inference(frame)
            
            # Handle visualization and publishing
            self.handle_results(result_img, lane_mask, drivable_mask, from_webcam)
            
        except Exception as e:
            self.get_logger().error(f"Frame processing error: {e}")
            # Fallback to dummy processing
            result_img, lane_mask, drivable_mask = self.dummy_inference(frame)
            self.handle_results(result_img, lane_mask, drivable_mask, from_webcam)
    
    def yolopv2_inference(self, frame):
        """Real YOLOPv2 inference"""
        t1 = time_synchronized()
        
        # Preprocess
        img0 = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
        img = letterbox(img0, self.img_size, stride=32)[0]
        
        # Convert to tensor
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)
        img_tensor = torch.from_numpy(img).to(self.device)
        
        if self.use_half_precision:
            img_tensor = img_tensor.half()
        else:
            img_tensor = img_tensor.float()
        
        img_tensor /= 255.0  # Normalize
        
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            [pred, anchor_grid], seg, ll = self.model(img_tensor)
        
        # Clean up input tensor immediately
        del img_tensor
        
        # Process predictions
        pred = split_for_trace_model(pred, anchor_grid)
        pred = non_max_suppression(pred, self.conf_threshold, self.iou_threshold, 
                                 classes=None, agnostic=False)
        
        # Process segmentation
        result_img = img0.copy()
        
        # Resize segmentation to match image
        with torch.no_grad():
            seg_resized = torch.nn.functional.interpolate(seg, size=(720, 1280), mode='bilinear')
            ll_resized = torch.nn.functional.interpolate(ll, size=(720, 1280), mode='bilinear')
            
            drivable_mask = torch.argmax(seg_resized, dim=1).squeeze().cpu().numpy().astype(np.uint8)
            lane_mask = torch.argmax(ll_resized, dim=1).squeeze().cpu().numpy().astype(np.uint8)
        
        # Clean up GPU memory aggressively
        del seg, ll, pred, anchor_grid, seg_resized, ll_resized
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Apply segmentation overlay
        show_seg_result(result_img, (drivable_mask, lane_mask), is_demo=True)
        
        t2 = time_synchronized()
        inference_time = t2 - t1
        
        return result_img, lane_mask, drivable_mask, inference_time
    
    def dummy_inference(self, frame):
        """Fallback dummy inference"""
        h, w = frame.shape[:2]
        
        # Create dummy masks
        lane_mask = np.zeros((h, w), dtype=np.uint8)
        drivable_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Simple lane simulation
        lane_mask[int(h*0.7):h, int(w*0.4):int(w*0.6)] = 1
        drivable_mask[int(h*0.5):h, int(w*0.2):int(w*0.8)] = 1
        
        # Create result image
        result_img = frame.copy()
        result_img[lane_mask == 1] = [0, 255, 255]  # Yellow lanes
        result_img[drivable_mask == 1] = [0, 255, 0]  # Green drivable area
        result_img = cv2.addWeighted(frame, 0.7, result_img, 0.3, 0)
        
        return result_img, lane_mask, drivable_mask
    
    def handle_results(self, result_img, lane_mask, drivable_mask, from_webcam=False):
        """Handle visualization and publishing"""
        # Show window if enabled
        if self.show_window and from_webcam:
            cv2.imshow('YOLOPv2 Clean Node', result_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                self.get_logger().info("User requested shutdown")
                self.destroy_node()
                return
        
        # Publish ROS image
        try:
            ros_image = self.bridge.cv2_to_imgmsg(result_img, 'bgr8')
            ros_image.header.stamp = self.get_clock().now().to_msg()
            ros_image.header.frame_id = 'camera'
            self.image_pub.publish(ros_image)
            
            # Publish pointclouds if enabled
            if self.publish_pointclouds:
                self.publish_pointclouds_func(lane_mask, drivable_mask)
                
        except Exception as e:
            self.get_logger().error(f"Publishing error: {e}")
    
    def publish_pointclouds_func(self, lane_mask, drivable_mask):
        """Publish PointCloud2 messages"""
        timestamp = self.get_clock().now().to_msg()
        
        # Lane pointcloud (red)
        lane_pc = self.mask_to_pointcloud(lane_mask, timestamp, 'camera', 
                                        [1.0, 0.0, 0.0], self.depth)
        self.lane_pc_pub.publish(lane_pc)
        
        # Drivable area pointcloud (green)
        drivable_pc = self.mask_to_pointcloud(drivable_mask, timestamp, 'camera',
                                            [0.0, 1.0, 0.0], self.depth)
        self.drivable_pc_pub.publish(drivable_pc)
    
    def mask_to_pointcloud(self, mask, timestamp, frame_id, color, depth):
        """Convert segmentation mask to PointCloud2"""
        # Find valid pixels
        y_coords, x_coords = np.where(mask == 1)
        
        if len(y_coords) == 0:
            return self.create_empty_pointcloud(timestamp, frame_id)
        
        # Subsample for performance
        step = max(1, len(y_coords) // self.max_points)
        y_coords = y_coords[::step]
        x_coords = x_coords[::step]
        
        # Camera parameters (simplified)
        h, w = mask.shape
        fx, fy = 500.0, 500.0
        cx, cy = w / 2.0, h / 2.0
        
        # Convert to 3D points
        points = []
        for x_img, y_img in zip(x_coords, y_coords):
            x_3d = (x_img - cx) * depth / fx
            y_3d = -((y_img - cy) * depth / fy)  # Flip Y for ROS convention
            points.append([x_3d, y_3d, depth, color[0], color[1], color[2]])
        
        # Create PointCloud2
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
        
        # Pack data
        pc.data = bytearray()
        for point in points:
            pc.data.extend(struct.pack('ffffff', *point))
        
        return pc
    
    def create_empty_pointcloud(self, timestamp, frame_id):
        """Create empty PointCloud2"""
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
        """Clean shutdown"""
        self.get_logger().info("Shutting down YOLOPv2 Clean Node...")
        
        if hasattr(self, 'cap'):
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        super().destroy_node()


def main(args=None):
    """Main function"""
    rclpy.init(args=args)
    
    try:
        node = YOLOPv2CleanNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Node error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()