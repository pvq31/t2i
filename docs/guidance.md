# agent_check_pkl_v5.py 的 guidance 与 pkl 参数对应规则

本文档用于指导 `python3 agent_check_pkl_v5.py --guidance "..."` 的写法，让中文或英文 guidance 都能更稳定地转换为 pkl 中的数值修改。

核心原则：

1. 中文和英文都可以作为关键词；但要写清楚对象名和操作。
2. 每句话只描述一个对象和一个操作，避免“整体更合理一点”这类模糊表达。
3. 对象名称尽量与 pkl 里的 `name` 或 `type` 一致，例如 `television`、`table`、`plant`、`teddy`、`bookshelf`。
4. 对于 `--guidance` 中没有提到的 cube，脚本会尽量保持原值不变。
5. guidance 中的旋转角度使用角度制；写入 pkl 时脚本会转成弧度制。

---

## 1. `dims`: 物体尺寸 `[Width, Depth, Height]`

pkl 字段：

```json
"dims": [width, depth, height]
```

例子：

```json
"dims": [1.1, 0.35, 1.9]
```

含义：

- `1.1` 是 `width`：左右宽度。
- `0.35` 是 `depth`：前后深度。
- `1.9` 是 `height`：上下高度。

### 1.1 所有维度一起增大 / 减小

对应 pkl 参数：

```text
dims[0], dims[1], dims[2]
```

中文 guidance 写法：

- `把 XX 所有维度增大。`
- `把 XX 所有维度减小。`
- `放大 XX。`
- `缩小 XX。`
- `适度放大 XX。`
- `适度缩小 XX。`
- `XX 整体尺寸增大。`
- `XX 整体尺寸减小。`

English guidance 写法：

- `Increase all dimensions of XX.`
- `Decrease all dimensions of XX.`
- `Make XX larger.`
- `Make XX smaller.`
- `Enlarge XX moderately.`
- `Reduce XX moderately.`
- `Increase the overall size of XX.`
- `Decrease the overall size of XX.`

稳定关键词：

- 增大：`larger` / `bigger` / `increase` / `enlarge` / `grow` / `放大` / `增大` / `更大`
- 减小：`smaller` / `reduce` / `decrease` / `shrink` / `tiny` / `缩小` / `减小` / `更小`

说明：

- `larger` / `enlarge` / `放大` 通常对应约 `1.12x`。
- `smaller` / `reduce` / `缩小` 通常对应约 `0.88x`。
- 如果写 `Increase all dimensions of television.` 或 `放大 television。`，会修改 `dims[0]`、`dims[1]`、`dims[2]`。

### 1.2 只修改 Width

对应 pkl 参数：

```text
width = dims[0]
```

中文 guidance 写法：

- `增大 XX 的 width 维度。`
- `减小 XX 的 width 维度。`
- `增大 XX 的宽度。`
- `减小 XX 的宽度。`
- `把 XX 加宽。`
- `把 XX 变窄。`

English guidance 写法：

- `Increase XX along the width dimension.`
- `Decrease XX along the width dimension.`
- `Increase the width of XX.`
- `Decrease the width of XX.`
- `Make XX wider.`
- `Make XX narrower.`

稳定关键词：

- 增大 width：`wider` / `widen` / `broader` / `increase width` / `more width` / `width dimension` / `更宽` / `加宽` / `变宽` / `宽度`
- 减小 width：`narrower` / `less wide` / `decrease width` / `reduce width` / `更窄` / `变窄` / `缩窄` / `宽度`

### 1.3 只修改 Depth

对应 pkl 参数：

```text
depth = dims[1]
```

中文 guidance 写法：

- `增大 XX 的 depth 维度。`
- `减小 XX 的 depth 维度。`
- `增大 XX 的深度。`
- `减小 XX 的深度。`
- `让 XX 更深。`
- `让 XX 更浅。`

English guidance 写法：

- `Increase XX along the depth dimension.`
- `Decrease XX along the depth dimension.`
- `Increase the depth of XX.`
- `Decrease the depth of XX.`
- `Make XX deeper.`
- `Make XX shallower.`

稳定关键词：

- 增大 depth：`deeper` / `increase depth` / `more depth` / `depth dimension` / `更深` / `加深` / `深度`
- 减小 depth：`shallower` / `reduce depth` / `less deep` / `decrease depth` / `更浅` / `变浅` / `深度`

### 1.4 只修改 Height

对应 pkl 参数：

```text
height = dims[2]
```

中文 guidance 写法：

- `增大 XX 的 height 维度。`
- `减小 XX 的 height 维度。`
- `增大 XX 的高度。`
- `减小 XX 的高度。`
- `把 XX 加高。`
- `把 XX 变矮。`

English guidance 写法：

- `Increase XX along the height dimension.`
- `Decrease XX along the height dimension.`
- `Increase the height of XX.`
- `Decrease the height of XX.`
- `Make XX taller.`
- `Make XX shorter.`

稳定关键词：

- 增大 height：`taller` / `higher` / `more tall` / `increase height` / `more height` / `height dimension` / `更高` / `加高` / `变高` / `高度`
- 减小 height：`shorter` / `lower` / `less tall` / `decrease height` / `reduce height` / `更矮` / `变矮` / `降低高度` / `高度`

推荐写法：

```bash
--guidance "Increase the bookshelf along the width dimension. Increase the bookshelf along the height dimension."
```

```bash
--guidance "增大 bookshelf 的宽度。增大 bookshelf 的高度。"
```

### 1.5 按倍数直接计算参数

如果 guidance 明确写“变为现在的多少倍 / 缩小为现在的多少”，则直接按当前参数乘倍率计算。
这个计算规则适用于所有参数，不局限于 `dims`、`width`、`depth`、`height`。

核心规则：

- `dims = [width, depth, height]`
- `dims[0]` 对应 `width`
- `dims[1]` 对应 `depth`
- `dims[2]` 对应 `height`

例子：只修改 `bed` 的高度，也就是只修改 `dims[2]`

```text
原始:
bed.dims = [width, depth, height]

如果 guidance 是：
"bed 的 height 减小为现在的 1/2"

则：
new_bed.dims[2] = old_bed.dims[2] * 1/2
```

```text
如果 guidance 是：
"bed 的 height 增加为现在的 1.2 倍"

则：
new_bed.dims[2] = old_bed.dims[2] * 1.2
```

注意：

- 上面两个例子都只修改 `dims[2]`。
- `dims[0]` 和 `dims[1]` 保持不变。
- 同理：
  - `width` 按倍数修改时，只改 `dims[0]`
  - `depth` 按倍数修改时，只改 `dims[1]`
  - “所有维度变为现在的 1.2 倍”时，同时对 `dims[0]`、`dims[1]`、`dims[2]` 都乘 `1.2`
- 更一般地，其他参数也遵循同样规则：
  - 如果 guidance 是“某个参数减小为现在的 `1/2`”，就把该参数乘 `1/2`
  - 如果 guidance 是“某个参数增加为现在的 `1.2` 倍”，就把该参数乘 `1.2`
- 也就是说，“按倍数修改”本质上就是：

```text
new_parameter = old_parameter * multiplier
```

---

## 2. `x`: 前后平移

pkl 字段：

```json
"x": [value]
```

含义：

- `x` 数字增大：物体向前 / 靠近相机 / closer to viewer。
- `x` 数字减小：物体向后 / 远离相机 / away from camera。
- 也就是说：`x` 表示前后平移。

中文 guidance 写法：

- `XX 向前平移。`
- `XX 向后平移。`
- `XX 往前移动。`
- `XX 往后移动。`
- `XX 靠近相机。`
- `XX 远离相机。`

English guidance 写法：

- `Move XX forward.`
- `Move XX backward.`
- `Shift XX forward.`
- `Shift XX backward.`
- `Move XX closer to the camera.`
- `Move XX away from the camera.`

稳定关键词：

- 向前，`x` 增大：`move forward` / `shift forward` / `forward` / `toward the camera` / `closer to the camera` / `closer to viewer` / `前移` / `往前` / `向前` / `靠近相机`
- 向后，`x` 减小：`move backward` / `move back` / `shift back` / `backward` / `back` / `away from the camera` / `farther from the camera` / `后移` / `往后` / `向后` / `远离相机`

注意：

- pkl 中保存的是渲染坐标。
- Gradio 界面显示的 x 与 pkl 的 x 有固定偏移：

```text
Gradio_x = pkl_x + 6.0
pkl_x = Gradio_x - 6.0
```

### 2.1 按参考物体的 Depth 做前后平移

guidance 要求：

- `XX 向前平移 0.7 个 YY 的 Depth。`
- `XX 向后平移 0.7 个 YY 的 Depth。`
- `Move XX forward by 0.7 times the depth of YY.`
- `Move XX backward by 0.7 times the depth of YY.`

对应参数调整：

1. 根据通用计算规则，计算 `YY` 的 `Depth` 乘倍率后的数值：

```text
delta_x = YY_depth * multiplier
```

例如：

```text
delta_x = YY_depth * 0.7
```

2. 调整 `XX.x`：

- 向前平移：`XX.x = old_XX.x + delta_x`
- 向后平移：`XX.x = old_XX.x - delta_x`

说明：

- `0.7` 只是例子，实际按 guidance 中要求的数值计算。
- 这里使用的是 `YY` 的 `Depth = YY.dims[1]`。
- 本质规则仍然是先算倍率，再把这个值加到或减到目标参数上。

---

## 3. `y`: 左右平移

pkl 字段：

```json
"y": [value]
```

含义：

- `y` 数字增大：物体向右。
- `y` 数字减小：物体向左。
- 也就是说：`y` 表示左右平移。

中文 guidance 写法：

- `XX 向左平移。`
- `XX 向右平移。`
- `XX 往左移动。`
- `XX 往右移动。`
- `把 XX 放到 YY 的右侧。`
- `把 XX 放到 YY 的左侧。`

English guidance 写法：

- `Move XX left.`
- `Move XX right.`
- `Shift XX to the left.`
- `Shift XX to the right.`
- `Place XX to the right of YY.`
- `Place XX to the left of YY.`

稳定关键词：

- 向左，`y` 减小：`to the left` / `move left` / `shift left` / `leftward` / `left` / `toward the left` / `向左` / `左移` / `往左` / `左侧` / `左边`
- 向右，`y` 增大：`to the right` / `move right` / `shift right` / `rightward` / `right` / `toward the right` / `向右` / `右移` / `往右` / `右侧` / `右边`

推荐写法：

```bash
--guidance "Move the chair slightly to the left. Place the plant to the right of the television."
```

```bash
--guidance "chair 稍微向左平移。把 plant 放到 television 的右侧。"
```

### 3.1 按参考物体的 Width 做左右平移

guidance 要求：

- `XX 向左平移 1.2 个 YY 的 Width。`
- `XX 向右平移 1.2 个 YY 的 Width。`
- `Move XX left by 1.2 times the width of YY.`
- `Move XX right by 1.2 times the width of YY.`

对应参数调整：

1. 根据通用计算规则，计算 `YY` 的 `Width` 乘倍率后的数值：

```text
delta_y = YY_width * multiplier
```

例如：

```text
delta_y = YY_width * 1.2
```

2. 调整 `XX.y`：

- 向左平移：`XX.y = old_XX.y - delta_y`
- 向右平移：`XX.y = old_XX.y + delta_y`

说明：

- `1.2` 只是例子，实际按 guidance 中要求的数值计算。
- 这里使用的是 `YY` 的 `Width = YY.dims[0]`。

---

## 4. `z`: 上下平移 / 支撑关系

pkl 字段：

```json
"z": [value]
```

含义：

- `z` 数字增大：物体向上。
- `z` 数字减小：物体向下。
- 也就是说：`z` 表示上下平移。
- 普通落地物体通常 `z = 0.0`。
- 放在支撑物上的物体，`z` 应接近：

```text
support_z + support_height
```

### 4.1 直接上下移动

中文 guidance 写法：

- `XX 向上平移。`
- `XX 向下平移。`
- `XX 往上移动。`
- `XX 往下移动。`
- `抬高 XX。`
- `降低 XX。`

English guidance 写法：

- `Move XX up.`
- `Move XX down.`
- `Shift XX up.`
- `Shift XX down.`
- `Raise XX.`
- `Lower XX.`

稳定关键词：

- 向上，`z` 增大：`move up` / `shift up` / `upward` / `up` / `raise` / `lift` / `向上` / `上移` / `往上` / `抬高`
- 向下，`z` 减小：`move down` / `shift down` / `downward` / `down` / `lower` / `向下` / `下移` / `往下` / `降低`

### 4.1.1 按参考物体的 Height 做上下平移

guidance 要求：

- `XX 向上平移 0.5 个 YY 的 Height。`
- `XX 向下平移 0.5 个 YY 的 Height。`
- `Move XX up by 0.5 times the height of YY.`
- `Move XX down by 0.5 times the height of YY.`

对应参数调整：

1. 根据通用计算规则，计算 `YY` 的 `Height` 乘倍率后的数值：

```text
delta_z = YY_height * multiplier
```

例如：

```text
delta_z = YY_height * 0.5
```

2. 调整 `XX.z`：

- 向上平移：`XX.z = old_XX.z + delta_z`
- 向下平移：`XX.z = old_XX.z - delta_z`

说明：

- `0.5` 只是例子，实际按 guidance 中要求的数值计算。
- 这里使用的是 `YY` 的 `Height = YY.dims[2]`。

### 4.2 支撑关系，更推荐

中文 guidance 写法：

- `把 XX 放在 YY 上。`
- `把 XX 放在 YY 顶部。`
- `把 XX 放在 YY 桌面上。`
- `把 XX 放在 YY 的中央。`
- `把 XX 放在 YY 的后侧中央。`

English guidance 写法：

- `Place XX on top of YY.`
- `Put XX on top of YY.`
- `XX sits on YY.`
- `XX is placed on YY.`
- `Place XX at the center of YY.`
- `Place XX on the back-center of YY.`

稳定支撑关键词：

- `on top of`
- `sits on` / `sitting on`
- `placed on`
- `positioned on`
- `resting on` / `rests on`
- `atop`
- `放在` / `摆在` / `桌上` / `台上` / `顶上` / `顶部` / `上面`

常见支撑物类型：

- `table`
- `desk`
- `drawer`
- `bookshelf`
- `bed`
- `refrigerator`
- `oven`
- `microwave`

推荐写法：

```bash
--guidance "Place the television on top of the table. Place the teddy on top of the bookshelf."
```

```bash
--guidance "把 television 放在 table 上。把 teddy 放在 bookshelf 上。"
```


---

## 5. 相对空间关系：上/下、左/右、前/后

本节说明 guidance 中出现 `XX 放在 YY 上方/下方/左边/右边/前面/后面` 或 `XX 紧贴在 YY 上方/下方/左边/右边/前面/后面` 时，应该如何根据 pkl 参数计算和调整 `XX` 的位置。

区分规则：

- `放在`：表示相对位置正确即可，通常留出一个正的安全间距 `margin`，也就是 XX 要比 YY 更上/更下/更左/更右/更前/更后，但不要求接触。
- `紧贴在`：表示两个物体对应的边界要接触，也就是相关边界坐标严格相等。

推荐：

- 如果需要“接触/贴合/刚好放在边上”，就写 `紧贴在`。
- 如果只需要“位于某一侧/某一上方”，就写 `放在`。

坐标约定：

- `dims = [width, depth, height]`
- `width = dims[0]`：左右宽度，对应 `y` 方向占用范围。
- `depth = dims[1]`：前后深度，对应 `x` 方向占用范围。
- `height = dims[2]`：上下高度，对应 `z` 方向占用范围。
- `x` 增大表示向前，`x` 减小表示向后。
- `y` 增大表示向右，`y` 减小表示向左。
- `z` 增大表示向上，`z` 减小表示向下。

为方便描述，设：

```text
XX = 被移动物体
YY = 参考物体 / 支撑物体
XX_width  = XX.dims[0]
XX_depth  = XX.dims[1]
XX_height = XX.dims[2]
YY_width  = YY.dims[0]
YY_depth  = YY.dims[1]
YY_height = YY.dims[2]
YY_x = YY.x
YY_y = YY.y
YY_z = YY.z
```

注意：pkl 中 `x/y/z` 常以列表形式保存，例如 `"x": [-6.8]`，计算时取第一个值。

### 5.1 `XX 放在 YY 上方 / 上面 / 顶部 / on top of YY`

典型 guidance：

- `把 XX 放在 YY 上。`
- `XX 放在 YY 上方。`
- `XX 放在 YY 顶部。`
- `Place XX on top of YY.`
- `Put XX above YY.`
- `Place XX on YY.`

参数调整规则：

1. 根据 `YY_width / YY_depth / YY_x / YY_y` 计算 YY 的水平占用面积：

```text
YY_x_min = YY_x - YY_depth / 2
YY_x_max = YY_x + YY_depth / 2
YY_y_min = YY_y - YY_width / 2
YY_y_max = YY_y + YY_width / 2
```

2. 调整 `XX.x` 和 `XX.y`，让 XX 的中心落在 YY 的面积区间内。最稳定做法是直接对齐中心：

```text
XX.x = YY.x
XX.y = YY.y
```

如果 guidance 指定角落或相对位置，例如 `front-left of YY`，则在 YY 面积内偏移，但仍保持在区间内：

```text
XX.x ∈ [YY_x_min, YY_x_max]
XX.y ∈ [YY_y_min, YY_y_max]
```

3. 根据 `YY_height` 和 `YY_z` 计算 YY 顶部坐标：

```text
YY_top_z = YY_z + YY_height
```

4. 令 `margin_z > 0`，调整 `XX.z`，让 XX 在 YY 顶部之上，但不与 YY 接触：

```text
XX.z = YY_top_z + margin_z
```

说明：这里的 `z` 表示 cube 底面高度，因此 `XX.z > YY.z + YY_height` 表示 XX 的底面高于 YY 顶面。

推荐输出变化描述：

```text
参数名字：XX.x；原始参数:...；修改后的参数:YY.x；变化：对齐 YY 的 x，使 XX 位于 YY 面积内
参数名字：XX.y；原始参数:...；修改后的参数:YY.y；变化：对齐 YY 的 y，使 XX 位于 YY 面积内
参数名字：XX.z；原始参数:...；修改后的参数:YY.z + YY.height + margin_z；变化：变大，放到 YY 上方
```

#### 5.1.1 `XX 紧贴在 YY 上方 / 上面 / 顶部`

典型 guidance：

- `把 XX 紧贴在 YY 上。`
- `XX 紧贴在 YY 上方。`
- `Place XX tightly on top of YY.`
- `Place XX flush on YY.`

参数调整规则：

1. 仍然先根据 `YY_width / YY_depth / YY_x / YY_y` 计算 YY 的水平占用面积。

2. 调整 `XX.x` 和 `XX.y`，让 XX 的中心落在 YY 的面积区间内。最稳定做法仍然是：

```text
XX.x = YY.x
XX.y = YY.y
```

3. 根据 `YY_height` 和 `YY_z` 计算 YY 顶部坐标：

```text
YY_top_z = YY_z + YY_height
```

4. 调整 `XX.z`，让 XX 的底面与 YY 的顶面严格接触：

```text
XX.z = YY_top_z
```

也就是：

```text
XX_bottom_z = YY_top_z
```

### 5.2 `XX 放在 YY 下方 / 下面 / below YY`

典型 guidance：

- `把 XX 放在 YY 下方。`
- `XX 放在 YY 下面。`
- `Place XX below YY.`
- `Put XX under YY.`

参数调整规则：

1. 仍然先根据 YY 面积对齐 `XX.x` 和 `XX.y`：

```text
XX.x = YY.x
XX.y = YY.y
```

2. 根据 YY 底部坐标确定下方位置。若 `z` 表示底面高度，则 YY 底部为：

```text
YY_bottom_z = YY_z
```

3. 令 `margin_z > 0`，让 XX 的顶部低于 YY 底部：

```text
XX.z + XX_height < YY_bottom_z
```

最稳定写法：

```text
XX.z = YY_bottom_z - XX_height - margin_z
```

如果不希望 XX 落到地面以下，则需要再加地面约束：

```text
XX.z = max(YY_bottom_z - XX_height - margin_z, 0)
```

#### 5.2.1 `XX 紧贴在 YY 下方 / 下面`

典型 guidance：

- `把 XX 紧贴在 YY 下方。`
- `XX 紧贴在 YY 下面。`
- `Place XX tightly below YY.`
- `Put XX flush under YY.`

参数调整规则：

1. 先对齐 `XX.x` 和 `XX.y`：

```text
XX.x = YY.x
XX.y = YY.y
```

2. 根据 YY 底部坐标确定接触位置：

```text
YY_bottom_z = YY_z
```

3. 调整 `XX.z`，让 XX 的顶部与 YY 的底部严格接触：

```text
XX.z = YY_bottom_z - XX_height
```

### 5.3 `XX 放在 YY 左边 / 右边`

典型 guidance：

- `把 XX 放在 YY 左边。`
- `把 XX 放在 YY 右边。`
- `Place XX to the left of YY.`
- `Place XX to the right of YY.`

左右关系主要调整 `y`。

根据 YY 的左右边界：

```text
YY_left_y  = YY_y - YY_width / 2
YY_right_y = YY_y + YY_width / 2
```

根据 XX 的宽度，计算 XX 中心应该放在哪里：

```text
XX_half_width = XX_width / 2
margin = 0.05 或更大安全间距
```

放在左边：

```text
XX.y = YY_left_y - XX_half_width - margin
```

放在右边：

```text
XX.y = YY_right_y + XX_half_width + margin
```

为了让两者前后大致对齐，通常保持：

```text
XX.x = YY.x
```

如果 guidance 同时指定 `front-left` / `back-right`，则可额外调整 `x`。

推荐输出变化描述：

```text
参数名字：XX.y；原始参数:...；修改后的参数:...；变化：变小，向左，放到 YY 左侧
参数名字：XX.y；原始参数:...；修改后的参数:...；变化：变大，向右，放到 YY 右侧
```

#### 5.3.1 `XX 紧贴在 YY 左边 / 右边`

典型 guidance：

- `把 XX 紧贴在 YY 左边。`
- `把 XX 紧贴在 YY 右边。`
- `Place XX tightly to the left of YY.`
- `Place XX flush to the right of YY.`

参数调整规则：

根据 YY 的左右边界：

```text
YY_left_y  = YY_y - YY_width / 2
YY_right_y = YY_y + YY_width / 2
XX_half_width = XX_width / 2
```

紧贴在左边：

```text
XX.y = YY_left_y - XX_half_width
```

此时：

```text
XX_right_y = YY_left_y
```

紧贴在右边：

```text
XX.y = YY_right_y + XX_half_width
```

此时：

```text
XX_left_y = YY_right_y
```

为了让两者前后大致对齐，通常仍保持：

```text
XX.x = YY.x
```

### 5.4 `XX 放在 YY 前面 / 后面`

典型 guidance：

- `把 XX 放在 YY 前面。`
- `把 XX 放在 YY 后面。`
- `Place XX in front of YY.`
- `Place XX behind YY.`

前后关系主要调整 `x`。

根据 YY 的前后边界：

```text
YY_back_x  = YY_x - YY_depth / 2
YY_front_x = YY_x + YY_depth / 2
```

根据 XX 的深度，计算 XX 中心应该放在哪里：

```text
XX_half_depth = XX_depth / 2
margin = 0.05 或更大安全间距
```

放在前面：

```text
XX.x = YY_front_x + XX_half_depth + margin
```

放在后面：

```text
XX.x = YY_back_x - XX_half_depth - margin
```

为了让两者左右大致对齐，通常保持：

```text
XX.y = YY.y
```

推荐输出变化描述：

```text
参数名字：XX.x；原始参数:...；修改后的参数:...；变化：变大，向前，放到 YY 前方
参数名字：XX.x；原始参数:...；修改后的参数:...；变化：变小，向后，放到 YY 后方
```

#### 5.4.1 `XX 紧贴在 YY 前面 / 后面`

典型 guidance：

- `把 XX 紧贴在 YY 前面。`
- `把 XX 紧贴在 YY 后面。`
- `Place XX tightly in front of YY.`
- `Place XX flush behind YY.`

参数调整规则：

根据 YY 的前后边界：

```text
YY_back_x  = YY_x - YY_depth / 2
YY_front_x = YY_x + YY_depth / 2
XX_half_depth = XX_depth / 2
```

紧贴在前面：

```text
XX.x = YY_front_x + XX_half_depth
```

此时：

```text
XX_back_x = YY_front_x
```

紧贴在后面：

```text
XX.x = YY_back_x - XX_half_depth
```

此时：

```text
XX_front_x = YY_back_x
```

为了让两者左右大致对齐，通常仍保持：

```text
XX.y = YY.y
```

### 5.5 组合关系：front-left / front-right / back-left / back-right

如果 guidance 是组合位置，例如：

- `XX 放在 YY 左前方。`
- `XX 放在 YY 右前方。`
- `XX 放在 YY 左后方。`
- `XX 放在 YY 右后方。`
- `Place XX front-left of YY.`
- `Place XX front-right of YY.`
- `Place XX back-left of YY.`
- `Place XX back-right of YY.`

则同时调整 `x` 和 `y`：

```text
front  -> x 增大，使用 YY_front_x + XX_half_depth + margin
back   -> x 减小，使用 YY_back_x  - XX_half_depth + margin 的负方向
left   -> y 减小，使用 YY_left_y  - XX_half_width - margin
right  -> y 增大，使用 YY_right_y + XX_half_width + margin
```

例如 `XX 放在 YY 左前方`：

```text
XX.x = YY_front_x + XX_half_depth + margin
XX.y = YY_left_y - XX_half_width - margin
```

例如 `XX 放在 YY 右后方`：

```text
XX.x = YY_back_x - XX_half_depth - margin
XX.y = YY_right_y + XX_half_width + margin
```

### 5.6 相对关系关键词汇总

中文关键词：

- 上方 / 上面 / 顶部 / 放在 ... 上 / 桌上 / 台上
- 下方 / 下面 / 底部 / 放在 ... 下
- 左边 / 左侧 / 左方 / 左面
- 右边 / 右侧 / 右方 / 右面
- 前面 / 前方 / 左前方 / 右前方
- 后面 / 后方 / 左后方 / 右后方

English keywords:

- `on top of` / `above` / `atop` / `placed on`
- `below` / `under` / `beneath`
- `left of` / `to the left of` / `on the left of`
- `right of` / `to the right of` / `on the right of`
- `in front of` / `front of` / `ahead of`
- `behind` / `back of` / `at the back of`
- `front-left` / `front-right` / `back-left` / `back-right`

### 5.7 注意事项

- 如果 `XX 放在 YY 上`，不仅要调 `z`，也要调 `x/y`，否则视觉上可能不在 YY 面积内。
- 如果 `XX 放在 YY 左/右/前/后`，通常只需要调整一个主轴，但为了视觉对齐，另一个轴可对齐到 YY 的中心。
- 如果 YY 有旋转 `azimuth`，严格计算应使用旋转后的包围盒；简化规则默认使用未旋转的 axis-aligned 面积。
- 如果要求“只修改 guidance 中提到的 cube”，则只移动 XX；YY 作为参考物体不应被改动。

---

## 6. `azimuth`: 水平旋转角度

pkl 字段：

```json
"azimuth": [radian]
```

含义：

- pkl 中保存的是弧度制 `radian`。
- guidance 和 Gradio 界面使用角度制 `degree`。
- 逆时针旋转 90 度：`Azimuth + 90°`。
- 顺时针旋转 90 度：`Azimuth - 90°`。
- 写入 pkl 前必须把角度转成弧度。

换算公式：

```text
degree = radian * 180 / π
radian = degree * π / 180
```

例子：

```text
1.5707963267948966 * 180 / π = 90°
90° * π / 180 = 1.5707963267948966
-90° * π / 180 = -1.5707963267948966
```

方向例子：

```text
当前 azimuth = 0°
逆时针旋转 90° -> 新 azimuth = 0° + 90° = 90° = 1.5707963267948966 rad
顺时针旋转 90° -> 新 azimuth = 0° - 90° = -90° = -1.5707963267948966 rad
```

中文 guidance 写法：

- `XX 逆时针旋转 90 度。`
- `XX 顺时针旋转 90 度。`
- `把 XX 逆时针转动 60 度。`
- `把 XX 顺时针转动 45 度。`

English guidance 写法：

- `Rotate XX 90 degrees counterclockwise.`
- `Rotate XX 90 degrees clockwise.`
- `Turn XX 60 degrees counterclockwise.`
- `Turn XX 45 degrees clockwise.`

稳定关键词：

- 旋转动作：`rotate` / `turn` / `spin` / `旋转` / `转动` / `转`
- 逆时针，角度增加：`counterclockwise` / `counter-clockwise` / `anticlockwise` / `anti-clockwise` / `ccw` / `逆时针`
- 顺时针，角度减少：`clockwise` / `顺时针`
- 角度单位：`degrees` / `degree` / `deg` / `°` / `度`

注意：

- guidance 中写角度制即可。
- `agent_check_pkl_v5.py` 会将角度转成弧度写入 pkl。

---

## 7. `camera_data.global_scale`: 全局缩放

pkl 字段：

```json
"camera_data": {
  "global_scale": 1.0
}
```

含义：

- 控制所有 cube 的整体尺寸缩放倍数。
- `global_scale > 1.0`：所有 cube 整体放大。
- `global_scale < 1.0`：所有 cube 整体缩小。
- `global_scale = 1.0`：不额外缩放。

中文 guidance 写法：

- `所有物体整体放大。`
- `所有物体整体缩小。`
- `所有 cube 整体放大。`
- `所有 cube 整体缩小。`
- `把 global_scale 设置为 1.2。`
- `把 global_scale 设置为 0.9。`

English guidance 写法：

- `Scale all cubes up.`
- `Scale all cubes down.`
- `Scale all objects up.`
- `Scale all objects down.`
- `Increase the global scale.`
- `Decrease the global scale.`
- `Set global scale to 1.2.`
- `Set global scale to 0.9.`

稳定关键词：

- 目标：`global scale` / `global_scale` / `all cubes` / `all cube` / `all objects` / `整体` / `所有物体` / `全部物体` / `所有cube`
- 放大：`scale up` / `larger` / `increase` / `放大` / `增大`
- 缩小：`scale down` / `smaller` / `decrease` / `reduce` / `缩小` / `减小`

---

## 8. `camera_data.camera_elevation`: 相机俯仰角 / 仰角

pkl 字段：

```json
"camera_data": {
  "camera_elevation": 0.2059488517353309
}
```

含义：

- 单位是弧度制 `radian`。
- `0.2059488517353309` rad 约等于 `11.8°`。
- 值越大，相机越从上往下看。
- 值越小，相机越接近水平视角。
- text-to-pkl 默认使用 `5°` 作为平视近似。
- 正角度表示 downward view；负角度表示 upward view。

换算公式：

```text
degree = radian * 180 / π
radian = degree * π / 180
```

中文 guidance 写法：

- `增大相机俯仰角。`
- `减小相机俯仰角。`
- `提高相机视角。`
- `降低相机视角。`
- `把 camera_elevation 设置为 12 度。`

English guidance 写法：

- `Increase camera elevation.`
- `Decrease camera elevation.`
- `Use a higher camera view.`
- `Use a lower camera view.`
- `Set camera elevation to 12 degrees.`
- `Use a 20 degree downward angle.`
- `Use a 30 degree upward view.`

稳定关键词：

- `camera elevation`
- `camera angle`
- `higher camera`
- `lower camera`
- `higher view`
- `lower view`
- `俯仰`
- `仰角`
- `相机`
- `视角`

---

## 9. `camera_data.lens`: 相机焦距

pkl 字段：

```json
"camera_data": {
  "lens": 50.0
}
```

含义：

- 表示相机焦距。
- 单位通常是 mm。
- 相机焦距数值越大，视野越窄、看起来更“拉近”。
- 相机焦距数值越小，视野越宽、看起来更“拉远”。
- `lens` 越小，更容易让所有 cube 入框。
- `lens` 越大，也更容易出画。

中文 guidance 写法：

- `使用更广的相机视野。`
- `减小相机焦距。`
- `增大相机焦距。`
- `拉远相机以包含所有 cube。`
- `把 lens 设置为 35。`

English guidance 写法：

- `Use a wider camera view.`
- `Decrease the camera lens.`
- `Increase the camera lens.`
- `Zoom out to include all cubes.`
- `Set camera lens to 35.`

稳定关键词：

- `camera lens`
- `lens`
- `zoom`
- `wider camera view`
- `wider view`
- `视野`
- `焦距`
- `拉远`
- `拉近`

对应文字：

- 相机焦距增大 / 减小
- `Increase the camera lens.`
- `Decrease the camera lens.`

数值规则：

- 如果 guidance 写的是“相机焦距增大10”或“相机焦距减小10”，这里的 `10` 表示 `camera_data["lens"]` 的绝对数值变化量。
- 也就是说：
  - `相机焦距增大10` → `camera_data["lens"] = camera_data["lens"] + 10`
  - `相机焦距减小10` → `camera_data["lens"] = camera_data["lens"] - 10`
- 这里不是按百分比，也不是乘倍率。
- 例如原始 `lens = 35` 时，`相机焦距增大10` 的结果应是 `45`，不是 `40`。

参数修改规则：

- 面对修改要求的文字时，和其他参数一样，直接指导修改 pkl 中的对应参数。
- 这里对应的是：

```json
"camera_data": {
  "lens": 50.0
}
```

- 如果 guidance 要求“增大相机焦距”，则增大 `camera_data["lens"]`
- 如果 guidance 要求“减小相机焦距”，则减小 `camera_data["lens"]`
- 如果 guidance 要求“把 lens 设置为 35”，则直接把 `camera_data["lens"]` 改为 `35`

按倍数计算说明：

```text
如果 guidance 是：
"相机焦距减小为现在的 1/2"

则：
new_camera_data["lens"] = old_camera_data["lens"] * 1/2
```

```text
如果 guidance 是：
"相机焦距增加为现在的 1.2 倍"

则：
new_camera_data["lens"] = old_camera_data["lens"] * 1.2
```

注意：

- 如果 guidance 写的是“增大/减小 10”这种明确数值变化，则按绝对值加减处理，不走乘倍率。
- 如果 guidance 写的是“变为现在的 1.2 倍 / 1/2”，则仍然按乘倍率处理：

```text
new_parameter = old_parameter * multiplier
```

- 这个“按倍数”规则不是 `lens` 独有的，而是所有参数在写明倍率时都适用。

---

## 10. 推荐 guidance 模板

### 10.1 尺寸

```bash
--guidance "Make the television larger. Keep its width and depth within the table."
```

```bash
--guidance "适度放大 television，但保持 television 的 width 和 depth 不超过 table。"
```

```bash
--guidance "Increase the bookshelf along the width dimension. Increase the bookshelf along the height dimension."
```

```bash
--guidance "增大 bookshelf 的宽度。增大 bookshelf 的高度。"
```

### 10.2 平移

```bash
--guidance "Move the chair slightly to the left. Move the table backward."
```

```bash
--guidance "chair 稍微向左平移。table 向后平移。"
```

### 10.3 支撑关系

```bash
--guidance "Place the television on top of the table. Place the teddy on top of the bookshelf."
```

```bash
--guidance "把 television 放在 table 上。把 teddy 放在 bookshelf 上。"
```

### 10.4 相对关系

```bash
--guidance "Place the plant to the right of the television. Place the chair front-left of the table."
```

```bash
--guidance "把 plant 放到 television 的右侧。把 chair 放到 table 的左前方。"
```

### 10.5 旋转

```bash
--guidance "Rotate the office chair 90 degrees counterclockwise."
```

```bash
--guidance "office chair 逆时针旋转 90 度。"
```

### 10.6 相机

```bash
--guidance "Set camera elevation to 12 degrees. Set camera lens to 35."
```

```bash
--guidance "把 camera_elevation 设置为 12 度。把 lens 设置为 35。"
```

---

## 11. 完整示例

英文：

```bash
python3 agent_check_pkl_v5.py \
--scene-text "In the image, there is a television in the center placed on a low table. A chair is positioned slightly front-left of the table. A plant is placed to the right of the television. Against the left wall, there is a bookshelf with books and a teddy on it." \
--scene-pkl inference/saved_scenes/test4.12/example_test4.12_7_fixed.pkl \
-o inference/saved_scenes/test4.12/example_test4.12_7_fixed2.1.pkl \
--guidance "Place the television on top of the table. Enlarge the television moderately, but keep its width and depth within the table. Place the plant to the right of the television. Place the teddy on top of the bookshelf. Rotate the chair 90 degrees counterclockwise. Set camera elevation to 12 degrees." \
--max-repair-attempts 4
```

中文：

```bash
python3 agent_check_pkl_v5.py \
--scene-text "In the image, there is a television in the center placed on a low table. A chair is positioned slightly front-left of the table. A plant is placed to the right of the television. Against the left wall, there is a bookshelf with books and a teddy on it." \
--scene-pkl inference/saved_scenes/test4.12/example_test4.12_7_fixed.pkl \
-o inference/saved_scenes/test4.12/example_test4.12_7_fixed2.1.pkl \
--guidance "把 television 放在 table 上。适度放大 television，但保持 television 的 width 和 depth 不超过 table。把 plant 放到 television 的右侧。把 teddy 放在 bookshelf 上。chair 逆时针旋转 90 度。把 camera_elevation 设置为 12 度。" \
--max-repair-attempts 4
```

---

## 12. 不推荐写法

不推荐：

```bash
--guidance "电视调一下，桌子也看看，整体更合理一点"
```

原因：对象和操作都不明确，规则解析不稳定。

推荐：

```bash
--guidance "把 television 放在 table 上。把 plant 放到 television 的右侧。把 teddy 放在 bookshelf 上。"
```

---

## 13. 新增/删除物体规则

guidance要求：增加物体XX。对应参数调整：增加一个对应物体，物体类别根据文字需求设定，包含的参数类型和其他cube一样，数值根据其他cube的相对大小和位置来进行设置。

guidance要求：删除物体XX。对应参数调整：删除对应物体。

---

## 14. 新增/删除物体（详细参数规则）

本节在第 13 节基础上，给出更细的“文字 -> pkl 参数”规则。保持与现有规则一致：能确定则确定性修改，不能确定则保守处理并尽量不影响其他 cube。

### 14.1 guidance要求：增加物体XX

对应参数调整：增加一个对应物体，物体类别根据文字需求设定，包含的参数类型和其他cube一样，包括：`"name"`、`"type"`、`"dims"`、`"x"`、`"y"`、`"z"`、`"azimuth"`、`"bbox"`。数值根据其他cube的相对大小和位置来进行设置。

#### 14.1.1 新增条目的参数模板

新增一个 `subjects_data` 条目，字段结构与已有 cube 完全一致：

```json
{
  "name": "object_name",
  "type": "asset_type",
  "dims": [width, depth, height],
  "x": [value],
  "y": [value],
  "z": [value],
  "azimuth": [value_in_radian],
  "bbox": [(0, 0, 0, 0)]
}
```

字段约束：

1. `"name"`：优先使用 guidance 中的对象名；若与现有对象重名，则自动追加编号（如 `chair 2`）。
2. `"type"`：按对象名映射到可识别资产类型（与现有 `match_asset_type` 规则一致）；无法识别时可回退到最接近类型或 `Custom`。
3. `"dims"`：优先使用该类型的参考尺寸；再根据文字语义（larger/smaller、宽高深关键词）做比例调整。
4. `"x"`, `"y"`：优先按相对关系（left/right/front/behind/center）相对已有对象定位；无明确关系时使用场景中心附近的安全位置。
5. `"z"`：默认落地（`z=0`）；若有 “on top of/放在…上” 则放到支撑物顶部。
6. `"azimuth"`：默认为该类型常用朝向；若 guidance 指定旋转角度，则按角度转弧度写入。
7. `"bbox"`：与已有数据格式一致，保持占位结构（例如 `[(0, 0, 0, 0)]`）。

#### 14.1.2 新增后的约束检查

新增后应继续满足已有规则：

1. 不与已有 cube 严重重叠（必要时沿 `x/y` 微调）。
2. 尽量完整入画；若越界，优先缩小该新增物体或轻微平移。
3. 若文字指定支撑关系，必须满足接触/落在支撑面上。
4. 新增对象不应破坏已明确要求保持不变的对象布局。

#### 14.1.3 guidance 写法示例

中文：

```bash
--guidance "增加物体 plant。把 plant 放到 table 的右侧。"
```

英文：

```bash
--guidance "Add a plant. Place the plant to the right of the table."
```

### 14.2 guidance要求：删除物体XX

对应参数调整：删除对应物体。

#### 14.2.1 删除匹配规则

1. 优先按 `name` 精确匹配删除（例如 `bed 2` 只删除 `bed 2`）。
2. 若未命中精确 `name`，可按基础名/类型匹配（如 `remove bed`）。
3. guidance 未明确编号且命中多个对象时，建议显式写编号；否则按规则可删除所有同类匹配项。

#### 14.2.2 删除动作

1. 从 `subjects_data` 中移除匹配对象的完整条目。
2. 仅删除目标对象，不修改未命中的对象参数。
3. 若目标对象不存在，则不改动 pkl，并返回提示信息。

#### 14.2.3 guidance 写法示例

中文：

```bash
--guidance "删除物体 lamp。"
```

英文：

```bash
--guidance "Delete the lamp."
```
