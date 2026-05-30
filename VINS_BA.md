# VINS-Fusion BA 设计说明

本文档记录当前这条分支里，VINS-Fusion 的滑动窗口 BA 与局部地图点管理是如何协同工作的。当前实现不是“纯 VINS 原始版本”，而是：

- **BA 主体仍然是 VINS 的 inverse-depth 窗口优化**
- **点管理与点筛选改成 AirSLAM 风格**
- **只有 `Good mappoint` 才能进入窗口 BA**
- **`UnTriangulated / Good / Bad` 三态由 `Mappoint` 维护**

相关代码主要位于：

- [vins_estimator/src/estimator/estimator.cpp](vins_estimator/src/estimator/estimator.cpp)
- [vins_estimator/src/estimator/feature_manager.cpp](vins_estimator/src/estimator/feature_manager.cpp)
- [vins_estimator/src/estimator/feature_manager.h](vins_estimator/src/estimator/feature_manager.h)
- [vins_estimator/src/utility/mappoint.h](vins_estimator/src/utility/mappoint.h)
- [vins_estimator/src/utility/mappoint.cpp](vins_estimator/src/utility/mappoint.cpp)
- [vins_estimator/src/utility/localmap_frame.h](vins_estimator/src/utility/localmap_frame.h)

---

## 总体设计

当前系统分成两层：

1. **点生命周期与筛选层**
   - 负责新点创建、观测累计、状态迁移、坏点剔除。
   - 采用 AirSLAM 风格三态：
     - `UnTriangulated`
     - `Good`
     - `Bad`
   - 这一层由 `Mappoint` 和 `FeatureManager` 协同维护。

2. **窗口 BA 层**
   - 仍使用 VINS 原有的 sliding-window inverse-depth BA。
   - 优化变量仍然是 `FeaturePerId::estimated_depth`。
   - 但进入 BA 的前提变成：**对应特征必须有 `Good mappoint`**。

也就是说：

- **`Mappoint` 决定“谁能进 BA”**
- **`estimated_depth` 决定“BA 优化什么”**
- **两者职责已经分离**

---

## 关键约束

### 1. `Good` 不再依赖 `estimated_depth`

`Good` 的状态不由 `estimated_depth` 或 `reliable_depth` 决定。  
`estimated_depth` 只是一项 BA 数值状态，不是点状态机的控制条件。

### 2. `estimated_depth` 不再控制点状态

深度可以被 BA 优化、写回、清空，但它不会把 `Good` 点“打回去”或“抬上来”。  
点状态只由生命周期逻辑维护。

### 3. `Good` 点不再走 triangulate 分支

`triangulate(...)` 只服务于 `UnTriangulated` 的初始化阶段。  
已经是 `Good` 的点，不再重新三角化，不再靠深度重算来解释自己的状态。

### 4. outlier 只会变成 `Bad`

BA 后如果点不稳定，会：

- 标记 `Bad`
- 删除/清理其观测
- 不再通过重新三角化把它“救回来”

---

## 点生命周期

### `UnTriangulated`

新点默认状态。特点是：

- 先积累观测
- 只有当观测数满足要求时，才考虑首次初始化
- 不进入窗口 BA

### `Good`

稳定点状态。特点是：

- 只由生命周期逻辑晋升得到
- 进入窗口 BA 的唯一门禁
- 不再触发 triangulate
- BA 里使用该点对应的 `estimated_depth`

### `Bad`

坏点状态。特点是：

- 直接剔除
- 不进入 BA
- 不再通过 retriangulate 恢复

---

## `feature_manager` 的职责

`FeatureManager` 现在承担两类职责：

### A. 点管理

- 收集每帧观测
- 为 feature 维护 observer
- 为 feature 维护对应 `Mappoint`
- 维护 `local_frames` / `local_mappoints`

### B. BA 门禁

- `useFeatureForOptimization()` 只负责判断“这个 feature 能不能进窗口 BA”
- 现在的门禁核心是：
  - 对应 `Mappoint` 存在
  - 状态是 `Good`

### C. 深度维护

- `triangulate(...)` 继续给 `FeaturePerId::estimated_depth` 初始化
- `setDepth(...)` 接收 BA 优化后的结果并写回
- `clearDepth()` 只清空深度，不改点状态

---

## 当前点管理流程

### 1. 观测进入

前端每帧观测会被加入 `FeatureManager::feature`，同时 local map 里会建立或更新 `Mappoint`。

### 2. `UnTriangulated` 先积累观测

`updateLocalPoints()` 每帧会维护 observer：

- 增加观测帧信息
- 统计 `ObverserNum()`
- 识别坏点

### 3. 首次初始化

当点还是 `UnTriangulated` 且观测足够时：

- `triangulate(...)` 负责给 `estimated_depth` 一个初值
- `Mappoint` 完成首次位置同步
- 然后点状态晋升为 `Good`

### 4. `Good` 点进入窗口 BA

`useFeatureForOptimization()` 只放行 `Good mappoint` 对应的特征。  
进入 BA 的仍然是 VINS 的 inverse-depth 参数。

### 5. BA 回写

BA 完成后：

- 优化结果写回 `FeaturePerId::estimated_depth`
- `setDepth(...)` 更新深度状态
- 需要时同步回 local map 点的位置

### 6. 外点处理

如果 BA 后点不稳定：

- 该点会被标记为 `Bad`
- 观测会被删除
- 不再重新三角化

---

## 窗口 BA 结构

当前 window BA 保持 VINS 原来的主结构：

- 位姿 `Pose`
- 速度与偏置 `SpeedBias`
- 相机外参 `Extrinsic`
- 时间偏差 `td`
- 视觉点的 `inverse depth`

变化只发生在“谁能进 BA”这一层：

- 以前是“满足 VINS 深度逻辑的点”
- 现在是“对应 `Mappoint` 为 `Good` 的点”

所以当前 BA 可以概括成：

- **VINS 的优化模型**
- **AirSLAM 风格的点门禁**

---

## 关键接口

### `FeatureManager::triangulate(...)`

作用：

- 仅为 `UnTriangulated` 点初始化 `estimated_depth`

不做的事情：

- 不负责 `Good` 点重三角化
- 不负责点状态晋升

### `FeatureManager::updateLocalPoints(...)`

作用：

- 维护 observer
- 更新 local map 中的 `Mappoint`
- 识别 `Bad`
- 将满足条件的 `UnTriangulated` 点晋升为 `Good`

不做的事情：

- 不把 `estimated_depth` 当作点状态控制器
- 不让 `Good` 点进入重新三角化分支

### `FeatureManager::useFeatureForOptimization(...)`

作用：

- 决定 feature 是否进入窗口 BA

当前门禁：

- 只看 `Mappoint::IsGood()`

### `FeatureManager::setDepth(...)`

作用：

- 接收 BA 结果
- 将优化后的 inverse depth 写回 `FeaturePerId::estimated_depth`
- 如果深度失效，标记该 feature 失败并清理

---

## 当前设计结论

这条线现在的真实设计可以概括为一句话：

> **AirSLAM 风格的点生命周期管理 + VINS 的 inverse-depth sliding-window BA**

其中：

- `Mappoint` 负责“点的状态”
- `estimated_depth` 负责“BA 的数值状态”
- `Good` 负责“进不进 BA”
- `Bad` 负责“删不删”

这比“深度控制状态、状态又反过来控制深度”更清晰，也更接近你现在想要的维护方式。

---

## 流程图

```text
            前端观测
                |
                v
      +----------------------+
      |  FeatureManager      |
      |  维护观测 / observer |
      +----------------------+
                |
                v
      +----------------------+
      |  UnTriangulated     |
      |  累积观测            |
      +----------------------+
                |
        ObverserNum() > 2
                |
                v
      +----------------------+
      |  triangulate(...)    |
      |  初始化 estimated_depth |
      +----------------------+
                |
                v
      +----------------------+
      |  Mappoint::Good      |
      |  只作为 BA 门禁      |
      +----------------------+
                |
                v
      +----------------------+
      |  Window BA           |
      |  优化 inverse depth  |
      +----------------------+
                |
                v
      +----------------------+
      |  setDepth()          |
      |  写回 estimated_depth |
      +----------------------+
                |
                v
      +----------------------+
      |  outlier?            |
      +----------------------+
           |          |
           | 否       | 是
           |          v
           |     +-----------+
           |     | Mappoint  |
           |     | 变 Bad    |
           |     +-----------+
           v
     继续参与后续 BA
```

这个图对应当前真实实现的核心关系：

- **点状态**由 `Mappoint` 管
- **深度数值**由 `FeaturePerId::estimated_depth` 管
- **进入 BA 的门禁**只看 `Good`
- **BA 优化后**再把结果写回深度
- **坏点**直接变 `Bad`，不再回到三角化链路
