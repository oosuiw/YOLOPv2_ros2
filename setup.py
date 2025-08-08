from setuptools import find_packages, setup
import glob
import os

package_name = 'yolopv2_ros'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob.glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob.glob('config/*')),
        ('share/' + package_name + '/model', glob.glob('model/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sws',
    maintainer_email='msmw1023@chosun.kr',
    description='ROS2 wrapper for YOLOPv2 panoptic driving perception',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolopv2_node = yolopv2_ros.yolopv2_node:main',
            'yolopv2_wrapper = yolopv2_ros.yolopv2_wrapper:main',
            'yolopv2_clean_node = yolopv2_ros.yolopv2_clean_node:main',
        ],
    },
)
