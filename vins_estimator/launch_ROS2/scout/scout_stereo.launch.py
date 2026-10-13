import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config_file = os.path.join(
        get_package_share_directory("stir_slam"),
        "config", "scout", "scout_stereo_config.yaml",
    )

    return LaunchDescription([
        DeclareLaunchArgument("config_file", default_value=config_file),
        Node(
            package="stir_slam",
            executable="stir_slam_node",
            name="stir_slam",
            output="screen",
            parameters=[{"config_file": LaunchConfiguration("config_file")}],
        ),
    ])
