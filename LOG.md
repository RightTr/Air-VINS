# LOG

## 2026-05-22

- Migrated the deep feature frontend code from AirSLAM into `vins_estimator/src/featureTracker/deepFeature/`
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
