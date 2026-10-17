import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("stir_slam")
    config_file = os.path.join(share, "config", "d455", "d455_mono_imu_config.yaml")
    rviz_launch = os.path.join(share, "launch", "stir_slam_rviz.launch.py")

    return LaunchDescription([
        DeclareLaunchArgument("config_file", default_value=config_file),
        DeclareLaunchArgument("visualization", default_value="true"),
        Node(
            package="stir_slam",
            executable="stir_slam_node",
            name="stir_slam",
            output="screen",
            parameters=[{"config_file": LaunchConfiguration("config_file")}],
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(rviz_launch),
            condition=IfCondition(LaunchConfiguration("visualization")),
        ),
    ])
