#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate launch description for YOLOPv2 Clean Node"""
    
    # Package directory
    pkg_share = FindPackageShare('yolopv2_ros')
    
    # Default config file
    default_config = PathJoinSubstitution([pkg_share, 'config', 'yolopv2_clean.yaml'])
    
    # Launch arguments
    config_arg = DeclareLaunchArgument(
        'config',
        default_value=default_config,
        description='Path to YAML config file'
    )
    
    use_webcam_arg = DeclareLaunchArgument(
        'use_webcam',
        default_value='false',
        description='Use webcam (true) or ROS topic (false)'
    )
    
    webcam_id_arg = DeclareLaunchArgument(
        'webcam_id',
        default_value='0',
        description='Webcam device ID'
    )
    
    input_topic_arg = DeclareLaunchArgument(
        'input_topic',
        default_value='/camera/image_raw',
        description='Input image topic'
    )
    
    # YOLOPv2 Clean Node
    yolopv2_node = Node(
        package='yolopv2_ros',
        executable='yolopv2_clean_node',
        name='yolopv2_clean_node',
        output='screen',
        parameters=[
            LaunchConfiguration('config'),
            {
                'use_webcam': LaunchConfiguration('use_webcam'),
                'webcam_id': LaunchConfiguration('webcam_id'),
                'input_topic': LaunchConfiguration('input_topic'),
            }
        ],
        remappings=[
            # Add any topic remappings here if needed
        ]
    )
    
    return LaunchDescription([
        config_arg,
        use_webcam_arg,
        webcam_id_arg,
        input_topic_arg,
        yolopv2_node
    ])