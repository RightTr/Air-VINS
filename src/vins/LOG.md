# LOG

## 2026-05-22

- 从 AirSLAM 迁入深度特征前端代码到 `vins_estimator/src/featureTracker/deepFeature/`
- 迁入模型文件到仓库 `output/`
- 在 `FeatureTracker` 中加入 `deep_feature` 开关
- 开关开启时，用 `SuperPoint + PointMatcher` 替代原 KLT 光流前端
- 保持后端 `Estimator` 和 `FeatureManager` 接口不变
- 在 `Estimator::clearState()` 中补充前端状态清理
- 更新 `vins_estimator/CMakeLists.txt` 以接入 TensorRT 和深度前端源码
- 在示例配置中加入 `deep_feature: 0`
- 把 `vins_estimator` 中左右图读取和轨迹显示接口改成 OpenCV4 写法
