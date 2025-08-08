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
    config_file = PathJoinSubstitution([pkg_share, 'config', 'yolopv2_webcam.yaml'])
    
    # Launch arguments
    webcam_id_arg = DeclareLaunchArgument(
        'webcam_id',
        default_value='0',
        description='Webcam device ID'
    )
    
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=config_file,
        description='Path to config file'
    )
    
    # YOLOPv2 wrapper node with webcam
    yolopv2_node = Node(
        package='yolopv2_ros',
        executable='yolopv2_wrapper',
        name='yolopv2_webcam_node',
        output='screen',
        parameters=[
            LaunchConfiguration('config_file'),
            {
                'use_webcam': True,
                'webcam_id': LaunchConfiguration('webcam_id'),
            }
        ]
    )
    
    return LaunchDescription([
        webcam_id_arg,
        config_file_arg,
        yolopv2_node
    ])