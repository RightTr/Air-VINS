# BA_process — 将 AirSLAM 的 BA 思路整合到 VINS 滑动窗口 BA

本文档给出一套可行的设计与实施步骤，目标是在保留 VINS-Fusion 滑动窗口 BA 基础上，借鉴 AirSLAM 在直线（Line）表示与优化上的做法，增强线特征三角化、BA 表示一致性与鲁棒性，同时保证边缘化先验的一致性与稳定性。

概念要点（核心取舍）
- 保持 VINS 的求解框架（Ceres + 滑动窗口 + 边缘化）。
- 在内部使线的表示与 AirSLAM 更兼容：AirSLAM 常用 6D Plücker 存储 + 4D 局部更新；VINS 当前以 4D `orth` 存储并有 `LineOrthParameterization`。
  - 選项 A：保持 VINS 的 `orth` 存储，但在三角化与可视化路径彻底使用 Plücker 做几何计算，保证转换函数（orth↔plk）无损且数值稳定。
  - 選项 B（较侵入）：改为以 6D Plücker 存储（line_state 存 plk），但保留 4D 局部参数化用于 Ceres（同 AirSLAM 思路）。该方案便于直接重用 AirSLAM 的几何/优化技巧，但改动较大，需修改 `para_LineFeature` 大小与 `lineProjectionFactor` 接口。

## 现在 VINS 的点 BA 到底是怎么做的

VINS 的点 BA 不是“把所有地图点做一次全局最小二乘”，而是“在滑动窗口里，把最近若干帧的位姿、IMU 状态、外参、时间偏差、以及与这些帧相关的地图点一起放进 Ceres 联合优化”。

对每个点特征，`feature_manager` 会保存它在多个帧上的观测；在 `Estimator::optimization()` 里，满足优化条件的点会被选出来，给每个点分配一个参数块 `para_Feature[k]`，然后为它的每个观测构造投影残差因子，例如同相机两帧的 `ProjectionTwoFrameOneCamFactor`，或者双目/跨相机的投影因子。优化时，Ceres 同时调整：

- 当前窗口内每一帧的位姿 `para_Pose[i]`
- 速度与 IMU 偏置 `para_SpeedBias[i]`（如果启用 IMU）
- 相机外参 `para_Ex_Pose[i]`
- 时间偏差 `para_Td[0]`
- 点的深度参数 `para_Feature[k]`

也就是说，VINS 的点 BA 本质上是一个“视觉约束 + IMU 约束 + 边缘化先验”的联合非线性优化。

### 点特征是怎么进优化的

点特征不是直接以 3D 世界坐标进入优化，而是通常以逆深或等价的单变量形式进入。直观理解是：

1. 前端先跟踪到每一帧的像素点（双目时还可能有右目观测）。
2. 这些像素观测会被整理成特征轨迹，进入 `FeatureManager`。
3. 当某个点满足初始化/可优化条件后，它会得到一个深度参数。
4. 在 BA 中，这个深度参数和相机位姿一起，被投影因子约束住。

这里的核心不是“点本身必须是 3D 地图坐标”，而是“这个 3D 点在不同相机位姿下投影回图像后，是否和真实观测一致”。

## 视觉约束到底怎么加进去的

在 VINS 里，“视觉约束”其实就是一类因子残差。它不是一个抽象的词，而是具体表现为 Ceres 里的 `AddResidualBlock()`。

对于点特征，典型流程是：

- 前端输出归一化后的点坐标 `point`，同时保留像素相关信息（用于显示、时间补偿或其他派生量）。
- 后端拿这些观测，构造投影因子。
- 每个投影因子会根据当前状态预测：
  - 这个 3D 点经过位姿变换后，应该落到哪一个像素位置；
  - 再和实际观测像素做差，形成 2 维残差。

所以“视觉约束”是整个残差项的统称，它包含：

- 点的重投影约束
- 双目左右相机之间的几何约束
- 线特征的投影约束
- 还有可能带时间偏差补偿的视觉约束

如果看代码，视觉约束真正进入优化的地方，就是这些因子的 `Evaluate()`，以及 `optimization()` 中的 `problem.AddResidualBlock(...)`。

## 视觉约束和重投影有什么区别

这两个词经常被混用，但严格说不是一回事。

### 1. 重投影（reprojection）是一个几何动作

重投影的意思是：

- 已知一个 3D 点
- 已知相机位姿和内外参
- 把这个 3D 点投到图像平面上

得到一个预测像素位置。这个过程叫重投影。

它本身还不是优化，只是“从 3D 到 2D 的几何映射”。

### 2. 视觉约束是一个优化残差

当你把“重投影得到的预测像素”与“真实观测像素”做差，并把这个差作为目标函数的一部分时，它才变成了视觉约束。

也就是：

$$
r = z_{obs} - \pi(T, X)
$$

其中：
- $z_{obs}$ 是真实观测
- $\pi(T, X)$ 是根据当前状态 $T$ 和地图点 $X$ 算出来的重投影结果
- $r$ 是残差

所以可以简单记成：

- 重投影 = 算出“应该落在哪里”
- 视觉约束 = “应该落在哪里”和“实际落在哪里”之间的误差

### 3. 在 VINS 中，视觉约束不只等于点重投影

在 VINS 的实现里，视觉约束至少包括：

- 点的重投影残差
- 双目左右相机的残差
- 线的投影残差

所以如果你说“视觉怎么加入约束”，更准确的说法是：

> VINS 把视觉观测转换成一组投影残差因子，然后把这些因子加到 Ceres 中，与 IMU 因子和边缘化先验一起联合优化。

## 一个更具体的理解方式

可以把它理解成三层：

1. 几何层
   - 做重投影，算预测像素。

2. 残差层
   - 用预测像素减真实像素，得到点/线的误差。

3. 优化层
   - 把这些误差作为约束项加入 BA，求最小二乘解。

因此，“重投影”是视觉约束的计算方式之一，但“视觉约束”是更上层的优化概念。

高层整合策略
1. 统一表示和转换
   - 确保代码里存在 `orth_to_plk`、`plk_to_orth`、`buildLineFromPlucker` 等工具（VINS 已有），并对它们进行数值稳定性测试。
   - 选择 A 或 B：推荐优先用 A（最小改动），把三角化 / plane-intersection / 可视化都用 Plücker 做运算，参数仍以 `orth` 存入 BA。若后来发现性能或语义上更需要 6D 存储，再迁移到选项 B。

2. 三角化流程（与 AirSLAM 对齐）
   - 立体优先：若第一观测为 stereo 且几何条件满足，像 AirSLAM 那样分别对线的两个端点执行 stereo 三角化（左右端点分别 triangulate），再用这两个三角化端点构造 Plücker 并用 `buildLineFromEndpoints` 规整化。
   - 两帧平面交（plane intersection）：保留 VINS 当前的 plane-intersection fallback，但调整细节使其与 AirSLAM 的 `TriangulateByTwoFrames` 行为一致：对每个端点用相机中心与端点构造平面，计算两个相机平面的交集（直线），并用最小二乘或 SVD 进一步稳健化端点位置。
   - 稳定端点与 Inlier 收集：三角化后用 `fitLine3DFromWorldPoints()` 收集支持的投影点/线段并拟合端点（RANSAC 或基于距离门限），然后把最终端点（或 plucker）转换回 `orth` 写入 `para_LineFeature`（若选择 A）。

3. BA 因子与参数化
   - 保持 `lineProjectionFactor` 的数学结构（基于 orth→plk 的投影残差）。如果切换到 plk 存储，必须修改 `lineProjectionFactor` 使其接受 plk 并在 `Evaluate()` 中使用 local 4D update（或保留 4D parameterization 局部表示，同时参数变量为 6D 实际存储，需要桥接层）。
   - 对最小更改，保留 `LineOrthParameterization`，但在三角化/初始化处强制把线归一化到正交表示以减少表征差异。

4. 边缘化与先验一致性
   - 最大风险：改变线的存储或参数个数会破坏 `last_marginalization_info`（历史先验与当前参数块不匹配）。策略：
     - 在任何改变 line 参数维度或排列（或外参、td）时，丢弃现有 `last_marginalization_info`（set to nullptr）并在下一个窗口完整重建先验（稍有额外成本，但保证正确性）。
     - 对逐步演进：先做非侵入式改动（A 方案），保留先验兼容；验证后再考虑切换到 6D 存储（B 方案），切换点强制清空历史先验并运行长期测试。

5. 线 BA 的选择条件
   - 保持 `LINE_BA` 开关，优化中只将满足 `useLineForOptimization()` 的线加入 BA。
   - 为了降低求解成本，考虑：
     - 限制每个窗口最大线数量 `NUM_OF_LINE_F`（已存在）。
     - 对线观测较差（低角度、多重共线、短像素长度）的观测预过滤，避免带来虚假约束。

6. 雅可比与鲁棒性
   - 确保 `lineProjectionFactor` 有精确雅可比（当前实现已包含）并提供数值检测工具（check()）用于验证新三角化/表示下的雅可比。
   - 对初始值敏感的线投影残差，适当使用鲁棒损失（Huber）与门限策略（lineOutliersRejection）。

7. 可视化与持久化
   - 三角化成功的线应像 3D 点一样被持久化并发布（VINS 已实现 persistent_map_lines），建议保持该行为并在 triage 失败时保留历史显示作为调试辅助。

实施步骤（分阶段，便于验收）

阶段 0 — 准备与测试工具
- 在 `utility` 中完善并单元测试：`orth_to_plk`、`plk_to_orth`、`buildLineFromEndpoints`、`fitLine3DFromWorldPoints` 等工具，并增加数值稳定性/边界测试。使用小合成数据验证转换无损。

阶段 1 — 非侵入式改进（三角化 + 可视化对齐）
- 实现或调整 `triangulateLine()` 中的三角化流程使之严格对齐 AirSLAM 的 stereo-first + two-frame plane-intersection。（已有类似实现，可做微调）
- 在三角化成功后把 plucker 线转换为 `orth` 并写入 `line_feature.line_3d_world`，保持 BA 不变。
- 增加更多稳健化：端点 RANSAC / dist gating / 最小支持点数。
- 测试：比较原始 VINS 输出与新逻辑在同一数据集上的轨迹与线地图差异，验证 BA 不发散。

阶段 2 — 优化流程与性能
- 调整 `LINE_BA` 的阈值（`LINE_MIN_OBS`、chi2、信息矩阵 scale），限制 `NUM_OF_LINE_F`，测量求解时间开销。
- 评估是否需要把部分线只在局部窗口短期优化而不是长期保持在参数里以减轻边缘化负担。

阶段 3 — 可选：改为 6D Plücker 存储（较大改动）
- 目标：内部存储改为 6D plucker，参数变量仍使用 4D 局部更新（实现方式与 AirSLAM 一致）。
- 需要变更：
  - `para_LineFeature` 的内存布局与 `lineProjectionFactor` 接口。
  - `line_parameterization` 或引入 plk→orth 层作为桥接。
  - 在切换点清空 `last_marginalization_info`，并运行长时间稳定性测试。
- 只有在 Stage1/2 效果不满意时才建议。

阶段 4 — 回归与长期验证
- 对比轨迹/地图（RMSE）、线地图质量（端点复现）、BA 求解时间、内存占用。
- 在多场景（室内直线主导、室外稀疏纹理、运动模糊）下做长期运行测试，关注边缘化稳定性与漂移。

验收标准（最小可接受目标）
- 三角化策略更新后系统能稳定运行（无优化发散）。
- 线被正确三角化并在地图中可视化，且在打开 `LINE_BA` 时能提升回环/跟踪稳定性（或至少不显著降低定位精度）。
- 性能开销在可接受范围（求解时间相较基线增长可控，或通过参数约束后可恢复）。


风险与注意事项
- 边缘化先验与参数维度不匹配会导致严重错误，切换参数布局时务必清空/重建先验。
- 线特征在稠密共线环境中容易引入虚假约束（过多平行直线），建议在加入 BA 前做更严格的筛选与局部验证。


我可以：
- 把上述阶段 1 的变更直接在 `FeatureManager::triangulateLine()` 中做出（小改动优先），并提交 PR；或
- 先把 `orth<->plk` 的单元测试与数值验证代码加到 `vins_estimator/tests/`，确保基础工具可靠后再改 triagulation 流程。

你希望我先做哪个步骤？