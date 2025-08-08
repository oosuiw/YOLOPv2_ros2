#!/usr/bin/env python3

import sys
import os
sys.path.append('/home/sws/libraries/YOLOPv2')

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


class YOLOPv2ROSWrapper(Node):
    def __init__(self):
        super().__init__('yolopv2_ros_wrapper')
        
        # Parameters
        # Model parameters
        self.declare_parameter('model_path', 'yolopv2_ros/model/yolopv2.pt')
        self.declare_parameter('model_type', 'pytorch')
        self.declare_parameter('use_half_precision', True)
        self.declare_parameter('device', 'auto')
        
        # Visualization parameters
        self.declare_parameter('show_realtime_window', True)
        self.declare_parameter('window_resizable', True)
        self.declare_parameter('window_width', 800)
        self.declare_parameter('window_height', 600)
        
        # Input parameters
        self.declare_parameter('use_webcam', False)  # Config will override this
        self.declare_parameter('webcam_id', 0)
        self.declare_parameter('input_topic', '/camera/image_raw')  # Config will override this
        self.declare_parameter('output_image_topic', '/yolopv2/detection_image')
        self.declare_parameter('lane_pointcloud_topic', '/yolopv2/lane_pointcloud')
        self.declare_parameter('drivable_pointcloud_topic', '/yolopv2/drivable_pointcloud')
        self.declare_parameter('camera_frame_id', 'camera')
        self.declare_parameter('camera_fx', 500.0)
        self.declare_parameter('camera_fy', 500.0)
        self.declare_parameter('camera_cx', 320.0)
        self.declare_parameter('camera_cy', 240.0)
        self.declare_parameter('lane_depth', 5.0)
        self.declare_parameter('drivable_depth', 5.0)
        self.declare_parameter('max_points_per_cloud', 1000)
        
        # Get model parameters
        self.model_path = self.get_parameter('model_path').value
        self.model_type = self.get_parameter('model_type').value
        self.use_half_precision = self.get_parameter('use_half_precision').value
        self.device_param = self.get_parameter('device').value
        
        # Get input parameters
        self.use_webcam = self.get_parameter('use_webcam').value
        self.webcam_id = self.get_parameter('webcam_id').value
        self.input_topic = self.get_parameter('input_topic').value
        self.output_image_topic = self.get_parameter('output_image_topic').value
        self.lane_pointcloud_topic = self.get_parameter('lane_pointcloud_topic').value
        self.drivable_pointcloud_topic = self.get_parameter('drivable_pointcloud_topic').value
        self.camera_frame_id = self.get_parameter('camera_frame_id').value
        self.camera_fx = self.get_parameter('camera_fx').value
        self.camera_fy = self.get_parameter('camera_fy').value
        self.camera_cx = self.get_parameter('camera_cx').value
        self.camera_cy = self.get_parameter('camera_cy').value
        self.lane_depth = self.get_parameter('lane_depth').value
        self.drivable_depth = self.get_parameter('drivable_depth').value
        self.max_points_per_cloud = self.get_parameter('max_points_per_cloud').value
        self.show_realtime_window = self.get_parameter('show_realtime_window').value
        self.window_resizable = self.get_parameter('window_resizable').value
        self.window_width = self.get_parameter('window_width').value
        self.window_height = self.get_parameter('window_height').value
        
        # Initialize model
        self.model = None
        self.device = None
        self.initialize_model()
        
        # CV bridge
        self.bridge = CvBridge()
        
        # Publishers
        self.image_pub = self.create_publisher(Image, self.output_image_topic, 10)
        self.lane_pointcloud_pub = self.create_publisher(PointCloud2, self.lane_pointcloud_topic, 10)
        self.drivable_pointcloud_pub = self.create_publisher(PointCloud2, self.drivable_pointcloud_topic, 10)
        
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
    
    def setup_window(self, window_name):
        """Setup OpenCV window with configurable size and properties"""
        try:
            # Create named window
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL if self.window_resizable else cv2.WINDOW_AUTOSIZE)
            
            # Set window properties
            if self.window_resizable:
                # Enable resizing
                cv2.setWindowProperty(window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FREERATIO)
                
                # Set initial size if specified
                if self.window_width > 0 and self.window_height > 0:
                    cv2.resizeWindow(window_name, self.window_width, self.window_height)
                    self.get_logger().info(f'Window initialized: {self.window_width}x{self.window_height} (resizable)')
                else:
                    self.get_logger().info('Window initialized: auto size (resizable)')
            else:
                self.get_logger().info('Window initialized: fixed size (non-resizable)')
                
        except Exception as e:
            self.get_logger().error(f'Failed to setup window: {str(e)}')
    
    def initialize_model(self):
        """Initialize YOLOPv2 model from config parameters"""
        try:
            # Resolve model path
            if self.model_path.startswith('yolopv2_ros/'):
                # Try multiple possible locations
                possible_paths = [
                    # Source directory
                    os.path.join('/home/sws/workspace/tmr_ws/src/yolopv2_ros', self.model_path.replace('yolopv2_ros/', '')),
                    # Install directory
                    os.path.join('/home/sws/workspace/tmr_ws/install/yolopv2_ros/share/yolopv2_ros', self.model_path.replace('yolopv2_ros/', '')),
                ]
                
                model_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        model_path = path
                        break
                
                if model_path is None:
                    model_path = possible_paths[0]  # Use first path for error message
            else:
                # Absolute path
                model_path = self.model_path
            
            # Check if model file exists
            if not os.path.exists(model_path):
                self.get_logger().error(f'Model file not found: {model_path}')
                return False
            
            # Setup device
            if self.device_param == 'auto':
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(self.device_param)
            
            self.get_logger().info(f'Using device: {self.device}')
            self.get_logger().info(f'Loading model from: {model_path}')
            
            # Load model based on type
            if self.model_type == 'pytorch':
                self.load_pytorch_model(model_path)
            elif self.model_type == 'onnx':
                self.load_onnx_model(model_path)
            elif self.model_type == 'tensorrt':
                self.load_tensorrt_model(model_path)
            else:
                self.get_logger().error(f'Unsupported model type: {self.model_type}')
                return False
            
            self.get_logger().info('Model loaded successfully')
            return True
            
        except Exception as e:
            self.get_logger().error(f'Failed to initialize model: {str(e)}')
            return False
    
    def load_pytorch_model(self, model_path):
        """Load TorchScript YOLOPv2 model"""
        self.get_logger().info(f'Loading TorchScript model: {model_path}')
        
        try:
            # Load TorchScript model directly (like original demo.py)
            self.model = torch.jit.load(model_path)
            self.model = self.model.to(self.device)
            
            if self.use_half_precision and self.device.type != 'cpu':
                self.model.half()
            
            self.model.eval()
            self.get_logger().info('TorchScript YOLOPv2 model loaded successfully')
            
            # Warm up with proper memory cleanup
            if self.device.type != 'cpu':
                dummy_input = torch.zeros(1, 3, 640, 640).to(self.device).type_as(next(self.model.parameters()))
                with torch.no_grad():
                    _ = self.model(dummy_input)
                del dummy_input
                torch.cuda.empty_cache()
                
        except Exception as e:
            self.get_logger().error(f'Failed to load TorchScript model: {str(e)}')
            raise
    
    def load_onnx_model(self, model_path):
        """Load ONNX model"""
        self.get_logger().info(f'Loading ONNX model: {model_path}')
        # Implement ONNX loading logic here
        self.model = "dummy_onnx_model"  # Placeholder
    
    def load_tensorrt_model(self, model_path):
        """Load TensorRT model"""
        self.get_logger().info(f'Loading TensorRT model: {model_path}')
        # Implement TensorRT loading logic here
        self.model = "dummy_tensorrt_model"  # Placeholder
    
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
        # 실제 YOLOPv2 모델 inference
        if self.model is None or isinstance(self.model, str):
            self.get_logger().warn('Model not loaded, using dummy processing')
            return self.dummy_process_frame(frame)
        
        try:
            # Import from local utils (copied from YOLOPv2)
            from .utils.utils import time_synchronized, non_max_suppression, split_for_trace_model, \
                scale_coords, driving_area_mask, lane_line_mask, plot_one_box, show_seg_result, letterbox
            
            # Resize and preprocess
            img0 = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
            img = letterbox(img0, 640, stride=32)[0]
            
            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img_tensor = torch.from_numpy(img).to(self.device)
            img_tensor = img_tensor.half() if self.use_half_precision else img_tensor.float()  # uint8 to fp16/32
            img_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0
            
            if img_tensor.ndimension() == 3:
                img_tensor = img_tensor.unsqueeze(0)
            
            # Inference with strict memory management
            t1 = time_synchronized()
            with torch.no_grad():
                [pred, anchor_grid], seg, ll = self.model(img_tensor)
            t2 = time_synchronized()
            
            # Immediately free input tensor
            del img_tensor
            
            # Process predictions
            pred = split_for_trace_model(pred, anchor_grid)
            pred = non_max_suppression(pred, 0.3, 0.45, classes=None, agnostic=False)
            
            # Process detections (박스 그리기 비활성화 - 차선만 보이게)
            result_img = img0.copy()
            # 객체 검출 박스는 그리지 않음 (차선 세그멘테이션만 표시)
            
            # Process segmentation masks - match the resized image dimensions (720, 1280)
            with torch.no_grad():
                da_seg_mask = torch.nn.functional.interpolate(seg, size=(720, 1280), mode='bilinear')
                ll_seg_mask = torch.nn.functional.interpolate(ll, size=(720, 1280), mode='bilinear')
                
                da_seg_mask = torch.argmax(da_seg_mask, dim=1).squeeze().cpu().numpy()
                ll_seg_mask = torch.argmax(ll_seg_mask, dim=1).squeeze().cpu().numpy()
            
            # Aggressively clean up GPU memory
            del seg, ll, pred, anchor_grid
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Apply segmentation overlay
            show_seg_result(result_img, (da_seg_mask, ll_seg_mask), is_demo=True)
            
            # Use real segmentation masks
            lane_mask = ll_seg_mask.astype(np.uint8)
            drivable_mask = da_seg_mask.astype(np.uint8)
            
            # 디버깅: 검출된 픽셀 수 확인
            lane_pixels = np.sum(lane_mask == 1)
            drivable_pixels = np.sum(drivable_mask == 1)
            self.get_logger().info(f'Lane pixels: {lane_pixels}, Drivable pixels: {drivable_pixels}, Inference: {t2-t1:.3f}s')
            
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                self.get_logger().error('CUDA out of memory - falling back to CPU processing')
                # Try switching to CPU temporarily
                try:
                    self.device = torch.device('cpu')
                    self.model = self.model.cpu()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    result_img, lane_mask, drivable_mask = self.dummy_process_frame(frame)
                except Exception:
                    result_img, lane_mask, drivable_mask = self.dummy_process_frame(frame)
            else:
                self.get_logger().error(f'YOLOPv2 inference failed: {str(e)}')
                result_img, lane_mask, drivable_mask = self.dummy_process_frame(frame)
        except Exception as e:
            self.get_logger().error(f'YOLOPv2 inference failed: {str(e)}')
            result_img, lane_mask, drivable_mask = self.dummy_process_frame(frame)
    
        # Continue with visualization and publishing
        self.handle_results(result_img, lane_mask, drivable_mask)
        
    def dummy_process_frame(self, frame):
        """Fallback dummy processing"""
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
        
        return result_img, lane_mask, drivable_mask
    
    def handle_results(self, result_img, lane_mask, drivable_mask):
        
        # 시각화 창 표시
        if self.show_realtime_window:
            window_name = 'YOLOPv2 Webcam' if self.use_webcam else 'YOLOPv2 Camera'
            
            # 창이 처음 생성될 때만 설정
            if not hasattr(self, 'window_initialized'):
                self.setup_window(window_name)
                self.window_initialized = True
            
            cv2.imshow(window_name, result_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                if self.use_webcam:
                    self.destroy_node()
                    return
        
        # ROS 이미지 퍼블리시
        try:
            ros_image = self.bridge.cv2_to_imgmsg(result_img, 'bgr8')
            ros_image.header.stamp = self.get_clock().now().to_msg()
            ros_image.header.frame_id = self.camera_frame_id
            self.image_pub.publish(ros_image)
            
            # PointCloud2 퍼블리시
            self.publish_pointclouds(lane_mask, drivable_mask)
            
        except Exception as e:
            self.get_logger().error(f'Error publishing: {str(e)}')
    
    def publish_pointclouds(self, lane_mask, drivable_mask):
        timestamp = self.get_clock().now().to_msg()
        
        # Lane pointcloud (빨간색) - 차선 깊이 사용
        lane_pc = self.mask_to_pointcloud2(lane_mask, timestamp, self.camera_frame_id, [1.0, 0.0, 0.0], self.lane_depth)
        self.lane_pointcloud_pub.publish(lane_pc)
        
        # Drivable area pointcloud (초록색) - 주행 영역 깊이 사용
        drivable_pc = self.mask_to_pointcloud2(drivable_mask, timestamp, self.camera_frame_id, [0.0, 1.0, 0.0], self.drivable_depth)
        self.drivable_pointcloud_pub.publish(drivable_pc)
    
    def mask_to_pointcloud2(self, mask, timestamp, frame_id, color, depth=None):
        y_coords, x_coords = np.where(mask == 1)
        
        if len(y_coords) == 0:
            return self.create_empty_pointcloud2(timestamp, frame_id)
        
        # 카메라 파라미터 (config에서 가져옴)
        fx, fy = self.camera_fx, self.camera_fy
        cx, cy = self.camera_cx, self.camera_cy
        z_depth = depth if depth is not None else self.lane_depth
        
        points = []
        # 포인트 수를 줄여서 성능 향상
        step = max(1, len(y_coords) // self.max_points_per_cloud)
        
        for i in range(0, len(y_coords), step):
            x_img, y_img = x_coords[i], y_coords[i]
            x_3d = (x_img - cx) * z_depth / fx
            y_3d = -((y_img - cy) * z_depth / fy)  # Y축 뒤집기 (카메라 좌표계 보정)
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