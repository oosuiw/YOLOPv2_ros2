#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Package directory
    pkg_share = FindPackageShare('yolopv2_ros')
    
    # Launch arguments
    weights_path_arg = DeclareLaunchArgument(
        'weights_path',
        default_value='/home/sws/libraries/YOLOPv2/data/weights/yolopv2.pt',
        description='Path to YOLOPv2 weights file'
    )
    
    conf_threshold_arg = DeclareLaunchArgument(
        'conf_threshold',
        default_value='0.3',
        description='Confidence threshold for object detection'
    )
    
    iou_threshold_arg = DeclareLaunchArgument(
        'iou_threshold', 
        default_value='0.45',
        description='IoU threshold for NMS'
    )
    
    img_size_arg = DeclareLaunchArgument(
        'img_size',
        default_value='640',
        description='Input image size'
    )
    
    input_topic_arg = DeclareLaunchArgument(
        'input_topic',
        default_value='/camera/image_raw',
        description='Input camera topic'
    )
    
    output_topic_arg = DeclareLaunchArgument(
        'output_topic',
        default_value='/yolopv2/detection_image',
        description='Output detection image topic'
    )
    
    use_webcam_arg = DeclareLaunchArgument(
        'use_webcam',
        default_value='false',
        description='Use webcam instead of ROS topic'
    )
    
    webcam_id_arg = DeclareLaunchArgument(
        'webcam_id',
        default_value='0',
        description='Webcam device ID'
    )
    
    # YOLOPv2 node
    yolopv2_node = Node(
        package='yolopv2_ros',
        executable='yolopv2_node',
        name='yolopv2_node',
        output='screen',
        parameters=[{
            'weights_path': LaunchConfiguration('weights_path'),
            'conf_threshold': LaunchConfiguration('conf_threshold'),
            'iou_threshold': LaunchConfiguration('iou_threshold'),
            'img_size': LaunchConfiguration('img_size'),
            'input_topic': LaunchConfiguration('input_topic'),
            'output_topic': LaunchConfiguration('output_topic'),
            'use_webcam': LaunchConfiguration('use_webcam'),
            'webcam_id': LaunchConfiguration('webcam_id'),
        }]
    )
    
    return LaunchDescription([
        weights_path_arg,
        conf_threshold_arg,
        iou_threshold_arg,
        img_size_arg,
        input_topic_arg,
        output_topic_arg,
        use_webcam_arg,
        webcam_id_arg,
        yolopv2_node
    ])