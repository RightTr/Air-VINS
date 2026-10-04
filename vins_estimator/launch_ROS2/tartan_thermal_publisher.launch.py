from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    args = [
        DeclareLaunchArgument(
            "data_root",
            default_value="/mnt/Data/TartanThermal/TartanRGBT_dataset/day1/undistorted_images_all_cameras_20250822_115703",
        ),
        DeclareLaunchArgument("playback_rate", default_value="1.0"),
        DeclareLaunchArgument("start_index", default_value="0"),
        DeclareLaunchArgument("max_frames", default_value="0"),
        DeclareLaunchArgument("publish_imu", default_value="true"),
        DeclareLaunchArgument("loop", default_value="false"),
        DeclareLaunchArgument("stereo_stamp_tolerance", default_value="0.003"),
        DeclareLaunchArgument("subscriber_wait_timeout", default_value="10.0"),
    ]
    publisher = Node(
        package="stir_slam",
        executable="tartan_thermal_publisher",
        name="tartan_thermal_publisher",
        output="screen",
        parameters=[{
            "data_root": LaunchConfiguration("data_root"),
            "left_dir": "thermal_left_rect_8",
            "right_dir": "thermal_right_rect_8",
            "imu_dir": "zed_imu_raw",
            "image0_topic": "/cam0/image_raw",
            "image1_topic": "/cam1/image_raw",
            "imu_topic": "/imu0",
            "playback_rate": ParameterValue(LaunchConfiguration("playback_rate"), value_type=float),
            "start_index": ParameterValue(LaunchConfiguration("start_index"), value_type=int),
            "max_frames": ParameterValue(LaunchConfiguration("max_frames"), value_type=int),
            "publish_imu": ParameterValue(LaunchConfiguration("publish_imu"), value_type=bool),
            "loop": ParameterValue(LaunchConfiguration("loop"), value_type=bool),
            "stereo_stamp_tolerance": ParameterValue(
                LaunchConfiguration("stereo_stamp_tolerance"), value_type=float
            ),
            "subscriber_wait_timeout": ParameterValue(
                LaunchConfiguration("subscriber_wait_timeout"), value_type=float
            ),
        }],
    )
    return LaunchDescription(args + [publisher])
