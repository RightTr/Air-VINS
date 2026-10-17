"""Run TartanThermal stereo/IMU playback and STIR-SLAM together."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory("stir_slam")
    publisher_config = os.path.join(
        share, "config", "tartan_thermal", "tartan_thermal_publisher.yaml"
    )
    estimator_config = os.path.join(
        share, "config", "tartan_thermal", "tartan_thermal_stereo_imu_config.yaml"
    )
    args = [
        DeclareLaunchArgument(
            "data_root",
            default_value="/mnt/Data/TartanThermal/TartanRGBT_dataset/day1/undistorted_images_all_cameras_20250822_115703",
        ),
        DeclareLaunchArgument("config_file", default_value=estimator_config),
        DeclareLaunchArgument("playback_rate", default_value="1.0"),
        DeclareLaunchArgument("start_index", default_value="0"),
        DeclareLaunchArgument("max_frames", default_value="0"),
        DeclareLaunchArgument("loop", default_value="false"),
        DeclareLaunchArgument("stereo_stamp_tolerance", default_value="0.003"),
        DeclareLaunchArgument("subscriber_wait_timeout", default_value="10.0"),
        DeclareLaunchArgument("visualization", default_value="true"),
    ]
    publisher = Node(
        package="stir_slam",
        executable="tartan_thermal_publisher",
        name="tartan_thermal_publisher",
        output="screen",
        parameters=[publisher_config, {
            "data_root": LaunchConfiguration("data_root"),
            "playback_rate": ParameterValue(LaunchConfiguration("playback_rate"), value_type=float),
            "start_index": ParameterValue(LaunchConfiguration("start_index"), value_type=int),
            "max_frames": ParameterValue(LaunchConfiguration("max_frames"), value_type=int),
            "loop": ParameterValue(LaunchConfiguration("loop"), value_type=bool),
            "stereo_stamp_tolerance": ParameterValue(
                LaunchConfiguration("stereo_stamp_tolerance"), value_type=float
            ),
            "subscriber_wait_timeout": ParameterValue(
                LaunchConfiguration("subscriber_wait_timeout"), value_type=float
            ),
        }],
    )
    estimator = Node(
        package="stir_slam",
        executable="stir_slam_node",
        name="stir_slam",
        output="screen",
        parameters=[{
            "config_file": LaunchConfiguration("config_file"),
            "reliable_sensor_qos": True,
        }],
    )
    rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(share, "launch", "stir_slam_rviz.launch.py")
        ),
        condition=IfCondition(LaunchConfiguration("visualization")),
    )
    return LaunchDescription(args + [publisher, estimator, rviz])
