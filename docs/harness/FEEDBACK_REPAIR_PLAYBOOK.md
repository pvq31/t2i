# Harness Feedback Repair Playbook

本文档规定 harness 在“评估 pkl 并产生修改意见”之后，如何把问题机械地转成 pkl 字段修改。目标是让闭环变成：

```text
生成 pkl -> 评估 pkl -> 生成可执行 repair_plan.json -> 执行 repair_plan -> 再评估
```

自然语言建议只用于人类阅读；真正修改必须进入 `repair_plan.json` 的字段级 action。

## 1. 关系类问题

### `relation_left_mismatch`

问题：`A` 没有在 `B` 左侧。

执行：

```text
A.y = min(A.y, B.y - B.width/2 - A.width/2 - margin_y)
```

### `relation_right_mismatch`

问题：`A` 没有在 `B` 右侧。

执行：

```text
A.y = max(A.y, B.y + B.width/2 + A.width/2 + margin_y)
```

### `relation_front_mismatch`

问题：`A` 没有在 `B` 前方。

执行：

```text
A.x = max(A.x, B.x + B.depth/2 + A.depth/2 + margin_x)
```

### `relation_behind_mismatch`

问题：`A` 没有在 `B` 后方。

执行：

```text
A.x = min(A.x, B.x - B.depth/2 - A.depth/2 - margin_x)
```

### 复合短语

`front-right/front-left/back-right/back-left` 必须拆成两个 predicate 后联合满足：

```text
front-right -> in_front_of + right_of
front-left  -> in_front_of + left_of
back-right  -> behind + right_of
back-left   -> behind + left_of
```

实现要求：左右关系只能约束 `y`，不能覆盖已有前后关系的 `x`；前后关系只能约束 `x`，不能覆盖已有左右关系的 `y`。

## 2. 可见性类问题

### 单个或少数 cube 出画：`frame_clipping`

问题：cube 没有完整落在图片内。

执行：

```text
target.x = target.x + 0.35 * (image_center_x_world - target.x)
target.y = target.y + 0.35 * (image_center_y_world - target.y)
```

当前 pkl 坐标中，图片中心对应的布局中心为：

```text
image_center_x_world = DEFAULT_CENTER_X
image_center_y_world = DEFAULT_CENTER_Y
```

如果该 cube 同时有空间谓词约束，例如 `left_of/right_of/in_front_of/behind`，平移后必须 clamp 到谓词允许区间内，不能为了可见性破坏文字关系。

无论有几个 cube 出画，都必须先对每个出画 cube 尝试沿 `y` 轴向图片中心机械平移。原因是左右出画通常由 `y` 过大/过小导致，不能跳过 y 平移直接缩尺寸或只改相机。

如果某个 cube 的 `y` 被 `left_of/right_of` 边界卡住，允许尝试平移该 cube 参与的 y 关系组件，以保持相对左右关系不变。但如果这个关系组件包含 `centered` 对象，例如 `sofa is in the center`，不能移动整组破坏居中对象。

如果某个 cube 带有 `screen_depth_axis` 前后屏幕深度约束，不能用 `x` 向图片中心回拉来修复 clipping。原因是 `front-right/front-left/in front of` 的前景对象需要保持靠近图片底部，`back-right/back-left/behind` 的背景对象需要保持靠近图片顶部；直接把 `x` 拉回中心会破坏屏幕空间分级，导致同一层级对象前后不一致。此时应优先保持 `x`，使用 `y` 平移或相机修复。

如果 clamp 后没有任何实际位移，说明该 cube 已被文字关系卡在边界上；此时 fallback 到相机修复：

```text
camera.lens *= 0.80
camera.global_scale *= 0.90
```

说明：1-2 个 cube 出画时优先移动目标 cube，而不是缩小尺寸。缩小尺寸只用于 `size_implausible`、支撑面放不下等尺寸问题。

### 多个 cube 出画：`frame_clipping`

问题：3 个及以上 cube 出画，通常说明相机视野或整体布局不合适。

执行优先级：

```text
camera.lens *= 0.80
camera.global_scale *= 0.90
```

字段必须 clamp 到 harness 允许范围：

```text
lens in [18, 80]
global_scale in [0.4, 2.5]
```

如果再次评估仍出画，下一轮继续降低焦距或整体 scale；不要只重复生成自然语言建议。

## 3. 尺寸类问题

### `size_implausible`

问题：cube 尺寸明显偏离资产参考尺寸。

执行：

```text
target.dims = visual_target_dims(target.type)
```

### `size_suspicious`

问题：cube 尺寸轻微偏离参考尺寸。

执行：

```text
target.dims = 0.5 * current.dims + 0.5 * visual_target_dims(target.type)
```

### `support_width_oversize`

问题：支撑物顶部放不下目标物体的 width。

执行：

```text
target.dims[0] *= 0.92
```

### `support_depth_oversize`

问题：支撑物顶部放不下目标物体的 depth。

执行：

```text
target.dims[1] *= 0.92
```

### `screen_size_reasonableness`

问题：某个 cube 投影后的 bbox 尺寸相对 `asset_dimensions.json` 参考比例过大或过小，例如前景 cat/dog 因透视显得大于 sofa/lamp。

执行必须调用 harness screen-size 工具，不写自然语言让 LLM 猜：

```text
screen_size_ratio = density / median_density
```

`allowed_range` 由空间谓词机械决定：

```text
普通物体:                          [0.8, 1.2]
front / front-left / front-right:  [0.8, 1.6]
clearly front:                     [0.75, 1.8]
far / very far / extra-far front:  [0.7, 2.0]
back / back-left / back-right:     [0.55, 1.2]
clearly back:                      [0.45, 1.15]
far / very far / extra-far back:   [0.35, 1.1]
```

任何 `dims` 补偿都必须限制在 `asset_dimensions.json` 默认尺寸的 `[0.7x, 1.4x]` 内，避免 screen-size 工具把 cat/dog 等前景小物体缩到不合理体积。

修复顺序：

1. 先沿 `x` 调整目标 cube，过大则远离相机，过小则靠近相机。
2. 若目标 cube 有 `front/behind` screen-depth 约束，`x` 调整不能破坏 `bottom_gap/top_gap` 分级。
3. 再尝试 `camera.lens` / `camera.global_scale`，但不能导致任何 cube 出画，也不能让 screen-depth 更差。
4. 若前景对象仍过大且继续缩小会触碰默认尺寸 0.7x 下限，则先把不参与空间谓词边界的中性参考物体在 1.4x 上限内等比例放大，提高 median density，并写入 `_screen_size_dim_compensated = true`。
5. 最后才允许等比例补偿 `dims`，并写入 `_screen_size_dim_compensated = true`；补偿后的 `dims` 仍必须满足默认尺寸 `[0.7x, 1.4x]` 安全边界。
6. 如果对象已经到达 0.7x 下限但 ratio 仍偏高，不能继续突破物理尺寸下限；如果对象已经到达 1.4x 上限但 ratio 仍偏低，不能继续突破上限。
7. 无 front/back screen-depth 的 left/right 普通关系对象不能被 screen-size 工具单独沿 `x` 反复拉动；这类对象若仍有 screen-size 偏差，只能通过相机或 `[0.7x, 1.4x]` 范围内的 dims 补偿处理。
8. screen-depth 验证使用 `3px` 容差，避免像素级误差造成 repair loop 震荡。
9. 补偿后必须再次恢复 screen-depth，保证前/后关系仍成立。

带 `_screen_size_dim_compensated` 的对象不再由 `size_implausible`、`size_suspicious` 或 `volume_order` 修回原始尺寸，否则 repair loop 会在 screen-size 与尺寸表之间震荡。

## 4. 物理类问题

### `below_ground`

问题：cube 底面低于地面。

执行：

```text
target.z = 0
```

### `floating_object`

问题：非支撑物体漂浮。

执行：

```text
target.z = 0
```

如果 prompt 中存在 `support(A, B)`，则由支撑关系工具执行：

```text
A.z = B.z + B.height
A.x/y clamp 到 B 的 footprint 内
```

## 5. 相机类问题

### `camera_range`

问题：相机参数超出 harness 允许范围。

执行：

```text
camera.lens = clamp(camera.lens, 18, 80)
camera.global_scale = clamp(camera.global_scale, 0.4, 2.5)
camera.camera_elevation = clamp(camera.camera_elevation, -45deg, 45deg)
```

默认 text-to-pkl 视角由 prompt-camera 工具锁定为 `5deg` 平视近似。只有 prompt 明确写出 `20 degree downward angle/tilt/view` 或 `30 degree upward angle/tilt/view` 这类角度短语时，才把 prompt 中的 degree 转成 radian，并将 `camera_elevation` 强制设为对应的正/负角度。

## 6. 闭环要求

每一轮必须保存：

```text
round_xx/validation.json
round_xx/repair_plan.json
round_xx_scene.pkl
round_xx_scene.json
round_xx_scene.repair_actions.jsonl
```

`repair_plan.json` 必须包含与失败问题对应的 action。例：

```json
{"tool": "set_param", "object": "lamp", "param": "dims", "value": [0.276, 0.276, 1.104]}
```

如果某轮仍失败，但 `repair_actions.jsonl` 为空，或连续两轮 `validation.issues` 完全相同且参数无变化，应判定为 `stalled`，不能继续假装修复。

如果 `repair_plan.action_count > 0`，但执行后 `repair_actions.jsonl` 为空，必须立即报错或写入 `summary.json`：

```json
{
  "stalled": true,
  "stalled_reason": "planned_actions > 0 but applied_actions == 0"
}
```

这类情况说明 repair plan 的字段格式或工具分发有问题，不能进入下一轮。

## 7. 当前实现位置

- 复合关系联合投影：`harness_param_constraints.py::apply_predicates`
- A-F issue 到字段级动作：`harness_param_constraints.py::apply_feedback_issues`
- 输出组合 repair plan：`agent_opinion.py`
- 执行 repair plan：`agent_reverse.py --repair-plan`
- 空转检测：`run_scene_harness.py`
