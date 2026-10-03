from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    rviz_config = os.path.join(
        get_package_share_directory("stir_slam"), "config", "stir_slam_rviz_config_ros2.rviz"
    )
    return LaunchDescription([
        Node(
            package="rviz2",
            executable="rviz2",
            name="rvizvisualisation",
            output="log",
            arguments=["-d", rviz_config],
        )
    ])
