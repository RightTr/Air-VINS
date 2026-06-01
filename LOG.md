# LOG

## 2026-05-22

- Migrated the deep feature frontend code from AirSLAM into `deepFeature/`
- Migrated model files into the repo `output/`
- Added a `deep_feature` switch in `FeatureTracker`
- When enabled, replaced the original KLT optical flow frontend with `SuperPoint + PointMatcher`
- Kept the backend `Estimator` and `FeatureManager` interfaces unchanged
- Added frontend state cleanup in `Estimator::clearState()`
- Updated `vins_estimator/CMakeLists.txt` to include TensorRT and the deep frontend sources
- Added `deep_feature: 0` to the example configs
- Updated image loading and track display code in `vins_estimator` to OpenCV4 APIs
- Added C++17 compatibility settings for `camera_models`, `loop_fusion`, and `global_fusion`
- Replaced remaining legacy OpenCV macros in `camera_models` with OpenCV4-compatible code so it builds in the current Docker environment
- Switched the EuRoC stereo and stereo-IMU deep feature configs to `deep_feature_matcher: 0` so they use LightGlue by default.

## NUS Thermal Stereo Updates

- Added a NUS thermal sequence publisher for `run_20250912_180927`.
- Added launch and config files for pure stereo thermal VINS using `/cam0/image_raw` and `/cam1/image_raw`.
- Added NUS thermal camera calibration files under `config/nus_thermal/`.
- Added stereo rectification before feature detection, using the NUS thermal intrinsics and left-right extrinsics.
- Updated the SLAM camera model to use the rectified virtual stereo intrinsics with zero distortion.
- Updated the rectified stereo extrinsics so VINS receives a consistent virtual stereo rig.
- Added optional IMU publishing from `realsense/imu/imu_synced.txt` on `/imu0`.
- Added a NUS thermal stereo-IMU config and launch file.
- Kept the raw D455f IMU frame as the VINS body frame and computed `body_T_cam0/body_T_cam1` for the rectified thermal virtual cameras.
- Fixed IMU replay ordering so every image frame is preceded by IMU samples through the first sample at or after the image timestamp.
- Added NUC/freeze handling for thermal stereo.
- When one thermal camera freezes, VINS falls back to monocular tracking using the other camera.
- When both thermal cameras freeze, visual updates are skipped and IMU propagation continues.
- Disabled stereo observations during NUC fallback so same-frame stereo triangulation is not used on frozen frames.
- Propagated the primary observation camera id through feature tracking, feature management, PnP, triangulation, reprojection checks, and visual residual construction.
- Ensured right-camera monocular fallback uses `body_T_cam1` instead of treating right images as left-camera observations.
- Enabled NUC handling in both NUS thermal stereo and stereo-IMU configs.
- Verified the updated package builds inside the `vinsfusion` Docker environment with `catkin_make`.
