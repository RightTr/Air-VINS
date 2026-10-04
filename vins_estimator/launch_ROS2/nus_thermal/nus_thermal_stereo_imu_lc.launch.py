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
        DeclareLaunchArgument("data_root", default_value="/mnt/Data/homebrew_thermal/Nus/run_20250902_162619"),
        DeclareLaunchArgument("config_file", default_value=os.path.join(share, "config", "nus_thermal", "nus_thermal_stereo_imu_config.yaml")),
        DeclareLaunchArgument("imu_file", default_value="realsense/imu/imu_synced.txt"),
        DeclareLaunchArgument("playback_rate", default_value="1.0"),
        DeclareLaunchArgument("start_index", default_value="0"),
        DeclareLaunchArgument("max_frames", default_value="0"),
        DeclareLaunchArgument("loop", default_value="false"),
        DeclareLaunchArgument("rectify_images", default_value="true"),
        DeclareLaunchArgument("rectify_alpha", default_value="0.0"),
        DeclareLaunchArgument("stereo_stamp_tolerance", default_value="0.005"),
        DeclareLaunchArgument("subscriber_wait_timeout", default_value="10.0"),
        DeclareLaunchArgument("visualization", default_value="true"),
    ]
    publisher = Node(
        package="stir_slam", executable="nus_thermal_publisher", name="nus_thermal_publisher", output="screen",
        parameters=[{
            "data_root": LaunchConfiguration("data_root"),
            "left_dir": "left_thermal/left_motion", "right_dir": "right_thermal/right_motion",
            "imu_file": LaunchConfiguration("imu_file"),
            "image0_topic": "/cam0/image_raw", "image1_topic": "/cam1/image_raw", "imu_topic": "/imu0",
            "publish_imu": True,
            "playback_rate": ParameterValue(LaunchConfiguration("playback_rate"), value_type=float),
            "start_index": ParameterValue(LaunchConfiguration("start_index"), value_type=int),
            "max_frames": ParameterValue(LaunchConfiguration("max_frames"), value_type=int),
            "loop": ParameterValue(LaunchConfiguration("loop"), value_type=bool),
            "rectify_images": ParameterValue(LaunchConfiguration("rectify_images"), value_type=bool),
            "rectify_alpha": ParameterValue(LaunchConfiguration("rectify_alpha"), value_type=float),
            "stereo_stamp_tolerance": ParameterValue(LaunchConfiguration("stereo_stamp_tolerance"), value_type=float),
            "subscriber_wait_timeout": ParameterValue(LaunchConfiguration("subscriber_wait_timeout"), value_type=float),
        }],
    )
    estimator = Node(package="stir_slam", executable="stir_slam_node", name="stir_slam", output="screen",
                     parameters=[{"config_file": LaunchConfiguration("config_file")}])
    loop_fusion = Node(package="loop_fusion", executable="loop_fusion_node", name="loop_fusion", output="screen",
                       parameters=[{"config_file": LaunchConfiguration("config_file")}])
    rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share, "launch", "stir_slam_rviz.launch.py")),
        condition=IfCondition(LaunchConfiguration("visualization")),
    )
    return LaunchDescription(args + [publisher, estimator, loop_fusion, rviz])
