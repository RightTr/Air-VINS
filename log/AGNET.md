# LocalMap 交接说明

## 目标
当前这条线的目标，是把系统从“VINS 窗口特征外挂 local point”收成 **AirSLAM 风格的稳定地图点层**，同时保留 **VINS 的 IMU-centered sliding-window BA** 作为主后端。

核心策略：

- 地图点先稳定
- stable 点再进入 window BA
- BA 结果再回写地图点
- tracking 先投影地图点，命中复用 id，失败才新建点
- 不迁移 mapline

---

## 当前实现状态

### 1. `localmap_frame.h`
`local_frame.h` 已改名为 `localmap_frame.h`，类型名仍然是 `LocalFrame`。

`LocalFrame` 现在是一个 **frame-owned local map 容器**，不是简单快照：

- 保存每帧的点槽
- 保存 feature id 映射
- 保存 track count
- 预留了 `line_observations` 入口，后面可以接线

相关文件：

- `vins_estimator/src/utility/localmap_frame.h`
- `vins_estimator/src/estimator/feature_manager.h`

---

### 2. `Mappoint` 状态机
`Mappoint` 已经收成了三层语义：

- `candidate`
- `stable`
- `retired`

另外还加了稳定性证据：

- 连续观测计数
- 重投影误差 EMA
- descriptor 距离 EMA
- 稳定确认计数

当前设计目标不是“够数就进 BA”，而是：

- 多视角持续重观测
- descriptor 稳定
- 几何误差长期低

满足这些条件后，点才真正变成 `stable`。

相关文件：

- `vins_estimator/src/utility/mappoint.h`
- `vins_estimator/src/utility/mappoint.cpp`

---

### 3. `FeatureManager` 的 local map 管理
`FeatureManager` 现在维护两层：

- `deque<LocalFrame> local_frames`
- `map<int, MappointPtr> local_mappoints`

它负责：

- push / pop local frame
- 给点加 / 删 observer
- 绑定 / 解绑 frame-owned slot
- 刷新 descriptor
- 导出可用于 tracking 的稳定点候选
- 更新点状态：candidate / stable / retired

重点逻辑在 `updateLocalPoints()`：

- 新点先作为 `candidate`
- 只有多视角观测质量足够，才升成 `stable`
- 失去支持就 `retired`
- 不是滑窗一滑就删点，删的是 observer，不是健康地图点本体

另外新增了更严格的 BA 入口：

- `getStableLocalMappointForBA()`

这个函数只让真正稳定的点进入 window BA。

相关文件：

- `vins_estimator/src/estimator/feature_manager.cpp`
- `vins_estimator/src/estimator/feature_manager.h`

---

### 4. Tracking 顺序
`FeatureTracker` 现在的顺序是：

1. 先投影 local stable map points
2. 命中就复用已有 id
3. 失败才生成新 candidate 点

也就是：

- 先找地图点
- 再找新点

这条顺序是为了减少同一结构反复生新 id，降低室外抖动。

相关文件：

- `vins_estimator/src/featureTracker/feature_tracker.cpp`

---

### 5. Window BA
`Estimator::optimization()` 里现在：

- 普通 VINS 点特征照旧
- stable local mappoint 才进入强约束
- 使用 `LocalPointReprojectionFactor`
- candidate 不进强 BA
- BA 后会回写 local map point

而且 BA 用的是更严格的入口：

- `getStableLocalMappointForBA()`

也就是说：

- 地图先稳
- BA 再用
- BA 结果反过来更新地图点

相关文件：

- `vins_estimator/src/estimator/estimator.cpp`
- `vins_estimator/src/factor/local_point_reprojection_factor.h`

---

## 当前关键约束

当前策略已经明确，不要再改成“点还没稳就先靠 BA 拉稳”。

稳定点必须满足：

- 连续多视角重观测
- descriptor 稳定
- 几何残差长期低
- observer 数达到最低门槛

BA 只吃：

- stable 点

不吃：

- candidate 点

---

## 当前不做的事

这条线里明确 **不迁移 mapline**。

线特征相关逻辑现在不是主目标，不要把 localmap 改成 mapline 系统。

---

## 当前风险点

这部分还没编译验证，所以下一个 agent 要优先检查：

1. `getStableLocalMappointForBA()` 是否在 `feature_manager.h/.cpp` 都完整声明/实现
2. `Mappoint` 新增的质量字段是否和序列化、构造函数一致
3. `LocalPointReprojectionFactor` 是否还能正常被 Ceres 接受
4. `updateLocalPoints()` 里对 stable / candidate 的判断是否过严，导致 BA 入口一直是 0
5. 现有 `getStableLocalMappoint()` / `getLocalMappointDepthPrior()` 和新的 BA 入口有没有语义冲突

---

## 建议的下一步

如果继续做这条线，优先顺序是：

1. 编译检查 localmap 相关改动
2. 打日志确认：
   - stable 点数量
   - `local point reprojections`
   - tracker 的 id 复用率
3. 如果室外还抖，继续收紧 stable 晋升条件，而不是放宽 BA

---

## 结论

现在 localmap 已经从“外挂缓存”改成了一个 **地图点驱动 tracking + VINS window BA** 的局部地图层。

当前语义是：

- 地图先稳
- stable 才进 BA
- BA 再回写地图点
- 不迁移 mapline
