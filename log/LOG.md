# LOG

## Current Work

- Refactoring `FeatureManager` into a thin coordinator that owns shared tracking state and window state.
- Splitting point and line logic into `PointFeatureManager` and `LineFeatureManager` under `vins_estimator/src/manager/`.
- Keeping point-only and line-only algorithms in their own managers, while moving shared state queries and common policy checks to the top-level manager.
- Removing redundant wrapper macros and duplicate guards, and tightening the manager layer logic.
- Renaming the mapline observer API from `Obverser` to `Observer`.
- Moving `deep_feature` header ownership into `deepFeature/` and removing the old copy from `featureTracker/`.
- Stabilizing NUS thermal stereo-IMU handling, including NUC fallback, stereo rectification, and right-camera monocular fallback.
- Keeping `line_ba` as the single switch for line inference, line BA, and line visualization.

## Loop Closure Work

- Integrating VPRNet / EigenPlaces as the loop descriptor path.
- Updating ONNX export to use `640x480` input.
- Aligning the TensorRT model export path with the loop_fusion runtime.
- Adding VPRNet loop-candidate debug logging.
- Adding a cooldown mechanism so the same old keyframe is not reused repeatedly in a short interval.
- Keeping VPRNet loop closure as an additional retrieval path on top of the existing keyframe verification flow.

## Frontend and Tracking Work

- Migrating the deep feature frontend into `deepFeature/` and keeping the `FeatureTracker` interface stable.
- Adding a `deep_feature` switch in the tracker frontend.
- When deep feature inference is enabled but fails, skipping that frame instead of falling back to KLT.
- Adjusting keyframe selection so it is less tightly coupled to windowed BA.
- Maintaining thermal stereo-IMU replay and feature tracking under the NUS thermal dataset.

## Recent Milestones

- Added loop-closure cooldown logic.
- Tightened the code logic in the manager and loop-fusion layers.
- Optimized the `FeatureManager` code structure.
- Removed deep feature code from `featureTracker`.
- Migrated files into the manager folder.
- Fixed the deep-feature loop-closure PGO bug.
- Added deep-feature-based PGO.
- Decoupled keyframe selection from windowed BA.
- Added stereo-IMU good-point initialization.
- Simplified the code structure around `ros_utils.h`.
- Added the AirSLAM-style keyframe selection strategy.
- Added TensorRT inference for the EigenPlaces / EightPlaces engine.
- Initialized NetVLAD support.
- Added mapline lifecycle handling from AirSLAM.
