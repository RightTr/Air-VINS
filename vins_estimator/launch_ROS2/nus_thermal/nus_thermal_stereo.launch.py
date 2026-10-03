from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    share = get_package_share_directory("stir_slam")
    args = [
        DeclareLaunchArgument("data_root", default_value="/mnt/Data/homebrew_thermal/Nus/run_20250912_180927"),
        DeclareLaunchArgument("config_file", default_value=os.path.join(share, "config", "nus_thermal", "nus_thermal_stereo_config.yaml")),
        DeclareLaunchArgument("playback_rate", default_value="1.0"),
        DeclareLaunchArgument("start_index", default_value="0"),
        DeclareLaunchArgument("max_frames", default_value="0"),
        DeclareLaunchArgument("loop", default_value="false"),
        DeclareLaunchArgument("rectify_images", default_value="true"),
        DeclareLaunchArgument("rectify_alpha", default_value="0.0"),
        DeclareLaunchArgument("visualization", default_value="true"),
    ]
    publisher = Node(
        package="stir_slam", executable="nus_thermal_publisher", name="nus_thermal_publisher", output="screen",
        parameters=[{
            "data_root": LaunchConfiguration("data_root"),
            "left_dir": "left_thermal/left_motion",
            "right_dir": "right_thermal/right_motion",
            "image0_topic": "/cam0/image_raw", "image1_topic": "/cam1/image_raw",
            "publish_imu": False,
            "playback_rate": ParameterValue(LaunchConfiguration("playback_rate"), value_type=float),
            "start_index": ParameterValue(LaunchConfiguration("start_index"), value_type=int),
            "max_frames": ParameterValue(LaunchConfiguration("max_frames"), value_type=int),
            "loop": ParameterValue(LaunchConfiguration("loop"), value_type=bool),
            "rectify_images": ParameterValue(LaunchConfiguration("rectify_images"), value_type=bool),
            "rectify_alpha": ParameterValue(LaunchConfiguration("rectify_alpha"), value_type=float),
        }],
    )
    estimator = Node(
        package="stir_slam", executable="stir_slam_node", name="stir_slam", output="screen",
        parameters=[{"config_file": LaunchConfiguration("config_file")}],
    )
    rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share, "launch", "stir_slam_rviz.launch.py")),
        condition=IfCondition(LaunchConfiguration("visualization")),
    )
    return LaunchDescription(args + [publisher, estimator, rviz])
