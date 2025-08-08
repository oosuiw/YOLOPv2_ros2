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
    
    # Config file path
    config_file = PathJoinSubstitution([pkg_share, 'config', 'yolopv2_camera.yaml'])
    
    # Launch arguments
    input_topic_arg = DeclareLaunchArgument(
        'input_topic',
        default_value='/image_raw',
        description='Input camera image topic'
    )
    
    camera_frame_arg = DeclareLaunchArgument(
        'camera_frame',
        default_value='camera',
        description='Camera frame ID'
    )
    
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=config_file,
        description='Path to config file'
    )
    
    # YOLOPv2 camera node
    yolopv2_camera_node = Node(
        package='yolopv2_ros',
        executable='yolopv2_wrapper',
        name='yolopv2_camera_node',
        output='screen',
        parameters=[
            LaunchConfiguration('config_file'),
        ]
    )
    
    return LaunchDescription([
        input_topic_arg,
        camera_frame_arg, 
        config_file_arg,
        yolopv2_camera_node
    ])