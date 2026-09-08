# VINS-Fusion 地图管理策略说明

本文档总结当前项目中点特征 `PointFeature` / 地标 `Mappoint` 的完整管理逻辑，包括：

- 特征如何被选入地图
- 特征和地标如何建立关联
- 双目三角化和多帧三角化分别何时发生
- 三角化后的 refine 如何进行
- 特征和地标如何删除
- anchor 如何选择、如何迁移
- NUC 场景下这套逻辑如何继续工作

本文描述的是当前代码状态下的实际行为，而不是理想化的论文版流程。

---

## 1. 核心结论

这套地图管理的核心思想可以概括成一句话：

> **feature id 负责“关联”，mappoint 负责“状态”，anchor 负责“参考系”。**

更具体地说：

- `PointFeatureId` 表示一条特征 track，保存该特征在多个 frame 上的观测。
- `Mappoint` 表示一个持久化的 3D 地标。
- 两者通过 `point_feature_id` 绑定。
- `anchor` 不是一个固定不变的帧，而是这条 track 当前最早还保留的观测。
- 三角化不是一次性的，后续会不断 refine、回写深度、迁移 anchor。

---

## 2. 数据结构与角色分工

### 2.1 `PointFeatureFrame`

`PointFeatureFrame` 是“**一帧里的一个观测单元**”。

它保存的不是某个相机的永久状态，而是某个 feature 在当前 frame 的观测结果。

它包含：

- `camera_id`
- `point`
- `pointRight`
- `uv`
- `uvRight`
- `is_stereo`

含义如下：

- `point`：当前相机归一化坐标或带深度的观测向量
- `pointRight`：同一帧右目的观测
- `is_stereo`：这一帧是否同时拿到了左右目

注意：

- **一个 `PointFeatureFrame` 代表一帧**
- **不是代表一个相机**
- stereo 只是这一帧里多了一份右目观测

这点很重要，因为后续多帧三角化的 observer 统计也是按 frame 记，不按 camera 数记。

### 2.2 `PointFeatureId`

`PointFeatureId` 是一条 track 的容器。

它保存：

- `point_feature_id`
- `start_frame`
- `point_feature_frame`
- `estimated_depth`
- `reliable_depth`
- `solve_flag`
- `depth_fail_count`

它的语义是：

- `start_frame`：这条 track 当前最早观测所在的 frame
- `point_feature_frame.front()`：当前 anchor observation
- `estimated_depth`：这条 track 在当前 anchor 下的深度
- `solve_flag`：
  - `0`：未稳定求解
  - `1`：已成功求解 / 深度可用
  - `2`：失败或外点，后续会被清理

### 2.3 `Mappoint`

`Mappoint` 是持久化 3D 地标。

它保存：

- 3D 位置
- 状态：`UnTriangulated / Good / Bad`
- observer 列表
- track count
- stale count
- descriptor

它的作用是：

- 保存“这个点在世界中的真实 3D 位置”
- 维护这个点还能不能继续用于优化
- 不直接等价于某一帧的观测

### 2.4 三者关系

可以把三者理解成：

- `PointFeatureFrame`：单帧观测
- `PointFeatureId`：一条观测链
- `Mappoint`：这条链对应的世界地标

关系是：

- `feature_id` 是桥
- `PointFeatureId` 负责把多帧观测串起来
- `Mappoint` 负责保存和更新 3D 状态

---

## 3. 特征是怎么被选进地图的

### 3.1 前端先产出 track

前端负责输出每一帧的特征集合。

在 deep 模式下：

- 时间上继续匹配历史特征
- 同帧左右目如果能匹配，也会合入同一帧观测
- 匹配成功的点复用原来的 `feature_id`

所以，前端并不是“每帧新建一批点”，而是：

- 尽量延续旧 id
- 只有新出现且允许建图的点才会分配新 id

### 3.2 `addPointFeatureCheckParallax`

`FeatureManager::addPointFeatureCheckParallax(...)` 是把前端输出的观测塞进 `pointFeature` 容器的入口。

它做的事情是：

1. 遍历当前帧的输入观测
2. 构造 `PointFeatureFrame`
3. 如果这个 `point_feature_id` 之前不存在，则新建 `PointFeatureId`
4. 如果已经存在，则在它的 `point_feature_frame` 后面追加一个新观测

这一步完成后：

- 一条 track 会随着时间不断增长
- 同一条 track 会包含多个 frame 的观测
- 如果某一帧是 stereo，则这一帧内部会同时保存左、右目观测

### 3.3 Parallax 门控

这个函数还会统计：

- 当前 track 数
- 新特征数
- 长 track 数
- 平均视差

然后用视差门控判断当前是否满足 keyframe / 新一轮三角化的条件。

它的作用不是“决定 feature 是否存在”，而是：

- 判断当前帧是否足够适合进入地图更新 / 三角化阶段

---

## 4. 特征和地标怎么绑定

### 4.1 绑定核心：`feature_id`

真正把特征和地标绑在一起的，不是 camera id，也不是 frame id，而是 `point_feature_id`。

规则是：

- 同一个 `point_feature_id` 对应同一个 `Mappoint`
- 前端只要继续沿用这个 id，后端就继续认为这是同一个 landmark

### 4.2 绑定动作：`updateLocalPoints`

`updateLocalPoints(...)` 会根据当前 `PointFeatureId` 去更新或创建 `Mappoint`。

它做的事情是：

1. 找到对应的 `Mappoint`
2. 清空旧 observers
3. 按当前 `point_feature_frame` 重建 observers
4. 更新 track count
5. 如果 `solve_flag == 2`，直接标坏
6. 如果已经是 `Good`，则用已有 `estimated_depth` 或 `Mappoint` 位置同步深度
7. 更新 window support，并清理 stale Mappoint

这一步的意义是：

- `PointFeatureId` 是“观测链”
- `Mappoint` 是“世界状态”
- 每轮都会把两者重新对齐一次

---

## 5. 三角化的时机

当前代码由 `triangulate(...)` 统一负责点三角化，`updateLocalPoints(...)` 不重复三角化。

### 5.1 进入窗口后先尝试

在关键帧或新帧进入后，系统会尽早尝试三角化，让能建的点尽快从 `UnTriangulated` 升到 `Good`。

### 5.2 优化阶段持续 refine

在后端优化过程里，会重复做：

- 在进入优化前尝试三角化
- 深度回写
- 地标状态更新

也就是说，三角化不是一次性初始化，而是一个持续 refine 的过程。

### 5.3 滑窗移动后还会再更新

当窗口滑动时，旧 anchor 可能被移出，新的 anchor 会顶上来。

这时系统还会重新解释深度，必要时再做 back-shift。

---

## 6. 双目三角化 vs 多帧三角化

### 6.1 双目三角化的定位

双目三角化只是一个“**初值种子**”。

它不是最终真值，只是：

- 如果视差足够靠谱
- 并且左右目的 y 对齐满足门限

就优先用双目快速得到一个 3D 初值。

### 6.2 当前双目 seed 的门控

当前逻辑是：

- 必须是 stereo anchor
- 必须是左目主观测
- 必须满足 y 方向门控
- 必须满足最小视差门控

如果任一条件不满足：

- 不做 stereo seed
- 直接进多帧 SVD fallback

### 6.3 为什么保留最小视差

最小视差的意义是：

- 提前过滤掉几何上过于退化的 stereo seed
- 少做一次没必要的三角化

现在已经统一要求：

- stereo 和 SVD 三角化后都要做深度检查

所以 `max disparity` 不再需要。

### 6.4 多帧三角化的定位

多帧三角化是主力。

它的含义是：

- 一条 track 在多个 frame 上都有观测
- 每个 frame 都参与约束
- 如果某一帧还是 stereo，则这一帧会额外贡献右目观测

这意味着：

- 不是只用左相机
- 也不是把左右目拆成两个独立时间链
- 而是按 frame 聚合后，再做统一的多视角求解

---

## 7. 多帧三角化具体怎么做

### 7.1 输入

输入是一条 `PointFeatureId` 的所有观测：

- 第一帧
- 第二帧
- 第三帧
- ...

每一帧可能是：

- 单目观测
- stereo 观测

### 7.2 构建方程

SVD 版本的多帧三角化会对每个 frame 构建线性方程。

如果某一帧是 stereo：

- 左目观测进方程
- 右目观测也进方程

因此：

- **observer 的基本单位是 frame**
- **方程数量可能因 stereo 而翻倍**

### 7.3 求解结果

SVD 得到的是一个全局 3D 点 `point_w`。

然后会：

- 把 `point_w` 投回 anchor 相机
- 得到当前 anchor 下的深度 `depth`
- 再做深度范围检查
- 再做重投影一致性检查

### 7.4 为什么 anchor 只是参考系

SVD 算出来的是世界点。

而 `estimated_depth` 只是这个世界点在当前 anchor 相机系下的表示。

所以：

- SVD 负责“点在哪里”
- anchor 负责“这个点在当前 track 的参考帧下有多深”

---

## 8. 深度检查怎么做

现在统一要求：

- **只要三角化出来，不管是双目还是 SVD，都要走深度检查**

### 8.1 双目 seed 后的检查

双目先算出一个初值之后：

- 立刻把它变成当前 anchor 下的深度
- 检查 `FEATURE_MIN_DEPTH / FEATURE_MAX_DEPTH`

如果深度不在范围内：

- 放弃这次 stereo seed
- 回退到多帧 SVD

### 8.2 SVD 后的检查

SVD 完成后：

- 先检查深度范围
- 再检查所有可见观测的重投影误差

如果任一项不通过：

- 这次三角化失败

### 8.3 为什么要这么做

因为三角化的结果本质上只是一个“候选 3D 点”。

必须通过后验检查，才能变成真正可用的地标。

---

## 9. refine 的方式

地图点不是“建出来就完事”，而是会持续 refine。

### 9.1 `triangulate`

这个函数的作用是：

- 遍历当前所有 track
- 对已有 `Mappoint` 的点，直接用 world point 反算当前深度
- 对没有 `Mappoint` 的点，尝试 stereo seed 或 SVD 三角化
- 成功后把点升成 `Good`

### 9.2 `updateLocalPoints`

`updateLocalPoints(...)` 会：

- 重新把 `PointFeatureId` 和 `Mappoint` 对齐
- 更新 observers
- 检查深度是否可靠
- 如果还能从 `Mappoint` 位置反推深度，就继续同步
- 再执行 stale pruning

### 9.3 `setDepth`

当 BA 或深度优化得到新的深度估计时：

- 如果深度有效，就更新 `estimated_depth`
- 如果无效，就增加 `depth_fail_count`
- 连续失败到阈值后，点会被标坏

### 9.4 refine 的本质

refine 不是重新发明一个点，而是：

- 让当前 `PointFeatureId`
- 和它对应的 `Mappoint`
- 在每轮窗口和优化后持续对齐

---

## 10. 特征删除与地标删除

删除不是单点动作，而是多级清理。

### 10.1 外点删除：`removePointFeatureOutlier`

当某些点被判为 outlier 时：

- 对应的 `PointFeatureId` 会从 track 集中移除
- 对应的 `Mappoint` 会被标成 `Bad`

这说明：

- 这条观测链已经不可信
- 地标不再继续用于后端优化

### 10.2 失败删除：`removeFailures`

当某个点的 `solve_flag == 2`：

- 直接删除

这类点通常是：

- 三角化失败
- 或者后续恢复失败

### 10.3 stale pruning：`pruneStaleLocalMappoints`

如果某个 `Mappoint` 已经不在当前活跃 `pointFeature` 里：

- 它不会立刻删除
- 会先增加 stale count
- 只要还有一定窗口支撑，就暂时保留

这一步很关键，因为它让系统在短时失观时不会立刻丢掉所有地标。

### 10.4 滑窗删除：`removeBack / removeFront / removePointFeatureBackShiftDepth`

窗口滑动时，最老的观测会被移除。

这时有三种情况：

1. 这条 track 还有后续观测
   - 删除最老观测
   - 新的第一帧自动变成新的 anchor

2. 这条 track 剩余观测太少
   - 直接删除这条 feature

3. 这条 track 还有足够观测
   - 用旧 anchor 的深度换算到新 anchor
   - 继续保留这条 track

这就是所谓的 **anchor 迁移**。

---

## 11. Anchor 的选择与迁移

### 11.1 anchor 不是单独存的变量

当前实现里，anchor 不是一个额外维护的索引，而是：

- `start_frame`
- `point_feature_frame.front()`

### 11.2 为什么默认用第一条 surviving observation

因为这套实现天然按时间顺序组织观测：

- 最早观测最适合作为 track 的 reference
- 滑窗时删掉最老观测后，新的最早观测自然接管

这样实现最简单，也最稳定。

### 11.3 anchor 如何迁移

假设一条 track 的观测序列是：

- 20
- 21
- 22
- 23

当 20 被滑窗移除后：

- 20 这条观测被删掉
- 21 自动成为新的 `front()`
- 所以 21 成为新的 anchor

再继续滑动：

- 21 出窗后，22 成为 anchor
- 22 出窗后，23 成为 anchor

也就是说：

> **anchor 是沿着 surviving observation 不断向前迁移的**

### 11.4 迁移时深度怎么处理

如果开启 back-shift depth：

- 先用旧 anchor 的深度恢复世界点
- 再投回新 anchor
- 得到新的深度

这不是重新三角化，而是：

- 同一个 3D 点
- 换一个参考相机系表达深度

### 11.5 是否可以选择“窗口内最好的一帧”当 anchor

理论上可以，但当前实现不这么做。

原因是：

- 当前索引语义已经建立在 `front()` 上
- 动态挑选最佳 anchor 会引入额外状态维护
- 容易破坏当前窗口滑动和深度回写逻辑

更合理的做法是：

- **结构 anchor** 继续保持第一 surviving observation
- **seed frame** 另外选最好的 stereo / 最稳的一帧

---

## 12. NUC 场景下地图管理如何工作

### 12.1 单相机接管

当某个热相机暂时 NUC：

- 另一相机会继续作为单目输入
- 这时系统不需要重建整个地图
- 已存在的 `feature_id` 和 `Mappoint` 会继续保留

### 12.2 继续追踪已有 landmark

剩余相机会：

- 继续做 descriptor matching
- 继续沿用旧 `feature_id`
- 后端继续拿这些 id 更新同一个 `Mappoint`

### 12.3 NUC 期间是否允许新地标

当前策略是允许的。

这意味着：

- NUC 期间也可以积累新的 track
- 这些 track 如果满足三角化条件，可以初始化成新 landmark
- 恢复双目后，这些点也可以继续作为约束使用

### 12.4 恢复后会发生什么

恢复后：

- NUC 期间建立的地标不会因为“曾经是单目”而被特殊对待
- 只要它们还没被标坏或删除，就继续参与优化
- 双目恢复后还可以进一步 refine

### 12.5 双相机同时失效

如果两边同时 NUC：

- 视觉测量暂时停止
- 系统只靠 IMU 传播

这不是新的地图状态机，而是视觉输入暂时中断。

---

## 13. 一条点从进入到删除的完整生命周期

可以把单个点的生命周期理解成下面这条链：

1. 前端检测并跟踪到一个 feature
2. `PointFeatureFrame` 被加入当前帧
3. `PointFeatureId` 持续追加多帧观测
4. 如果 stereo 合格，先尝试双目 seed
5. 如果双目不合格，走多帧 SVD
6. 三角化成功后，计算 `estimated_depth`
7. `Mappoint` 被创建或更新，并置为 `Good`
8. 后端优化继续 refine 深度和位置
9. 窗口滑动时，anchor 迁移，深度换基
10. 如果变成 outlier / solve 失败 / stale 太久
11. 地标被标坏或删除

---

## 14. 当前实现的简化规则

如果要把整套逻辑压缩成几条可执行规则，可以记成：

### 14.1 选特征

- 优先延续旧 `feature_id`
- 同帧 stereo 只是一帧里的多观测，不是多个独立 track

### 14.2 建关联

- `feature_id` 是唯一桥梁
- 只要 id 没变，后端就继续认为是同一个 landmark

### 14.3 三角化

- stereo 先试
- stereo 只做 `y + min disparity` 门控
- 不合格就直接走 SVD
- stereo 和 SVD 都要过深度检查

### 14.4 refine

- 有 `Mappoint` 就回写深度
- 没有 `Mappoint` 就尝试建点
- BA 后再统一更新深度

### 14.5 删除

- outlier 直接删
- solve fail 直接删
- stale 先缓删
- 窗口滑动时删掉最老观测，必要时迁移 anchor

---

## 15. 实现上的直觉总结

这套地图管理不是“点建出来就固定不变”，而是：

- **观测链不断延长**
- **anchor 不断迁移**
- **3D 点不断 refine**
- **地标在窗口里尽量活得更久**

所以它的目标不是追求“每个点第一次就完美”，而是：

> **尽可能让一个 landmark 在窗口里持续保持可用，并在滑窗、NUC、重投影和优化中不断自我修正。**
