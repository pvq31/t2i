# Prompt Relation Phrase Rules

本文档规定 text-to-pkl harness 允许用户使用的空间关系短语。解析器只按本文档中的中英文短语生成 predicate，不依赖大模型语义理解，也不把宽泛动词或介词自行推断成关系。

用户写 prompt 时必须使用本文档列出的说法。没有列出的自由表达可能不会被解析，或者只按对象数量规则处理。

## Predicate Schema

解析器输出以下 predicate：

```text
centered(subject)
left_of(subject, object)
right_of(subject, object)
in_front_of(subject, object)
behind(subject, object)
above(subject, object)
below(subject, object)
support(subject, object)
```

`subject` 是被放置/被约束的物体，`object` 是参照物体。

## English Rules

| Predicate | Required English Phrases |
| --- | --- |
| `centered(A)` | `A is in the center`, `A is at the center`, `A is in the center of the image`, `A is at the center of the image`, `A is centered`, `A is centered in the image`, `A centers the image`, `A anchors the image` |
| `left_of(A, B)` | `A is to the left of B`, `A is on the left of B`, `A is left of B`, `A is on the left side of B`, `A is lying on the left side of B`, `A is positioned to the left of B`, `A is located to the left of B`, `A is located to the left of the B`, `A stands left of B`, `A stands left of the B`, `A stands to the left of B`, `A sits left of B`, `A sits to the left of B`, `A sits to B's left`, `A sits to the B's left` |
| `right_of(A, B)` | `A is to the right of B`, `A is on the right of B`, `A is right of B`, `A is on the right side of B`, `A is lying on the right side of B`, `A is positioned to the right of B`, `A is located to the right of B`, `A is located to the right of the B`, `A stands right of B`, `A stands right of the B`, `A stands to the right of B`, `A sits right of B`, `A sits to the right of B`, `A sits to B's right`, `A sits to the B's right` |
| `in_front_of(A, B)` | `A is in front of B`, `A is front of B`, `A is positioned in front of B`, `A is located in front of B`, `A is front-right of B`, `A is front-left of B`, `A is located in front-right of B`, `A is located in front-left of B` |
| `behind(A, B)` | `A is behind B`, `A is at the back of B`, `A is in back of B`, `A is positioned behind B`, `A is located behind B`, `A is back-right of B`, `A is back-left of B`, `A is located in back-right of B`, `A is located in back-left of B` |
| `above(A, B)` | `A is above B`, `A is over B`, `A is higher than B` |
| `below(A, B)` | `A is below B`, `A is under B`, `A is beneath B`, `A is lower than B` |
| `support(A, B)` | `A is on top of B`, `A is atop B`, `A is placed on B`, `A is positioned on B`, `A is resting on B`, `A rests on B`, `A is sitting on B`, `A sits on B`, `A is on the countertop of B`, `A is on the surface of B`, `A is on the top of B` |

## Chinese Rules

| Predicate | Required Chinese Phrases |
| --- | --- |
| `centered(A)` | `A在画面中央`, `A位于画面中央`, `A在图像中央`, `A位于图像中央`, `A在中央`, `A位于中央`, `A居中` |
| `left_of(A, B)` | `A在B左边`, `A在B左侧`, `A位于B左边`, `A位于B左侧`, `A放在B左边`, `A放在B左侧` |
| `right_of(A, B)` | `A在B右边`, `A在B右侧`, `A位于B右边`, `A位于B右侧`, `A放在B右边`, `A放在B右侧` |
| `in_front_of(A, B)` | `A在B前面`, `A在B前方`, `A位于B前面`, `A位于B前方`, `A放在B前面`, `A放在B前方`, `A在B右前方`, `A位于B右前方`, `A在B左前方`, `A位于B左前方` |
| `behind(A, B)` | `A在B后面`, `A在B后方`, `A位于B后面`, `A位于B后方`, `A放在B后面`, `A放在B后方`, `A在B右后方`, `A位于B右后方`, `A在B左后方`, `A位于B左后方` |
| `above(A, B)` | `A在B上方`, `A位于B上方`, `A高于B` |
| `below(A, B)` | `A在B下方`, `A位于B下方`, `A低于B`, `A在B下面`, `A位于B下面` |
| `support(A, B)` | `A在B上`, `A放在B上`, `A放到B上`, `A摆在B上`, `A置于B上`, `A在B顶部`, `A放在B顶部`, `A在B台面上`, `A放在B台面上`, `A在B表面上`, `A放在B表面上` |

## Compound Phrases

复合短语会拆成两个 predicate：

```text
A is front-right of B -> in_front_of(A, B) + right_of(A, B)
A is front-left of B -> in_front_of(A, B) + left_of(A, B)
A is back-right of B -> behind(A, B) + right_of(A, B)
A is back-left of B -> behind(A, B) + left_of(A, B)
A在B右前方 -> in_front_of(A, B) + right_of(A, B)
A在B左前方 -> in_front_of(A, B) + left_of(A, B)
A在B右后方 -> behind(A, B) + right_of(A, B)
A在B左后方 -> behind(A, B) + left_of(A, B)
```

复合短语还会触发一个额外的机械距离规则：前后方向必须比普通 `in_front_of` / `behind` 更明显。默认值：

```text
compound_longitudinal_margin = 1.5
```

因此：

- `front-right` / `front-left`：同时满足左右关系，并要求 `back_edge(A) >= front_edge(B) + max(margin_fb, 1.5)`。
- `back-right` / `back-left`：同时满足左右关系，并要求 `front_edge(A) <= back_edge(B) - max(margin_fb, 1.5)`。
- 这个规则只加大前后方向距离，不额外加大左右方向距离。
- 目的是让 `front-right`、`front-left`、`back-right`、`back-left` 中的前后关系在 cube layout 中比普通斜向错位更明显。

## Screen-Space Front/Back Depth Levels

世界坐标的前后距离只能保证 3D layout 中 A 在 B 前方或后方，不能保证最终图像里 A 明显进入画面下半部分或上半部分。为了让 `front-right` / `front-left` / `back-right` / `back-left` 以及单纯 `in front of` / `behind` 的画面效果更可控，harness 应增加一层屏幕空间分级约束。

该约束使用投影后的 cube bbox 像素坐标，不使用世界坐标距离。图像坐标约定为左上角 `(0, 0)`，向右为 `x` 增大，向下为 `y` 增大。默认图像高度为 `1024px`。

```text
bbox_top(A)    = projected_bbox(A).min_y
bbox_bottom(A) = projected_bbox(A).max_y
top_gap(A)     = bbox_top(A)
bottom_gap(A)  = image_height - bbox_bottom(A)
```

执行顺序必须固定：

1. 先执行 predicate 空间关系。复合短语必须同时满足两个方向，例如 `front-right(A, B)` 必须先满足 `in_front_of(A, B)` 和 `right_of(A, B)`；`back-left(A, B)` 必须先满足 `behind(A, B)` 和 `left_of(A, B)`。单纯 `in front of` / `behind` 也必须先满足世界坐标前后关系。
2. 保持参照物 `B` 不动，投影 `B`，计算 `B` 到图片底部或顶部的像素距离。
3. 根据 prompt 强度查表，机械设置 `A` 的目标屏幕距离。
4. 只移动 `A` 的前后轴位置来接近目标；移动后必须重新验证 predicate 和 `full_cube_visible`。
5. 如果屏幕空间目标与完整可见性冲突，`full_cube_visible` 优先，A 停在仍完整可见的最近位置。

### Front / Front-Left / Front-Right

前方关系以图片底部为基准。A 必须比 B 更靠近图片底部：

```text
bottom_gap(A) <= bottom_gap(B) - delta_px
```

| Prompt strength | English examples | Chinese examples | delta_px | Target rule |
| --- | --- | --- | ---: | --- |
| slight | `slightly in front of B`, `slightly front-right of B`, `slightly front-left of B` | `稍微在B前方`, `稍微在B右前方`, `稍微在B左前方` | `80` | `bottom_gap(A) <= bottom_gap(B) - 80` |
| default | `A is in front of B`, `A is front-right of B`, `A is front-left of B` | `A在B前方`, `A在B右前方`, `A在B左前方` | `80` | `bottom_gap(A) <= bottom_gap(B) - 80` |
| clear | `A is clearly in front of B`, `A is clearly front-right of B`, `A is clearly front-left of B` | `A明显在B前方`, `A明显在B右前方`, `A明显在B左前方` | `260` | `bottom_gap(A) <= bottom_gap(B) - 260` |
| far | `A is far in front of B`, `A is far front-right of B`, `A is far front-left of B` | `A远在B前方`, `A远在B右前方`, `A远在B左前方` | `340` | `bottom_gap(A) <= bottom_gap(B) - 340` |
| extra-far | `A is very far in front of B`, `A is extra-far front-right of B`, `A is extra-far front-left of B` | `A非常远在B前方`, `A非常远在B右前方`, `A非常远在B左前方` | `420` | `bottom_gap(A) <= bottom_gap(B) - 420` |

如果 `bottom_gap(B) - delta_px` 小于安全留白，目标值应被夹到安全范围内：

```text
target_bottom_gap(A) = max(min_visible_bottom_gap_px, bottom_gap(B) - delta_px)
```

建议默认：

```text
min_visible_bottom_gap_px = 80
```

例如 `bottom_gap(B) = 500px` 且 prompt 是普通 `front-right`，则 `delta_px = 80px`，目标为：

```text
bottom_gap(A) <= 420px
```

### Back / Back-Left / Back-Right

后方关系以图片顶部为基准。A 必须比 B 更靠近图片顶部：

```text
top_gap(A) <= top_gap(B) - delta_px
```

| Prompt strength | English examples | Chinese examples | delta_px | Target rule |
| --- | --- | --- | ---: | --- |
| slight | `A is slightly behind B`, `A is slightly back-right of B`, `A is slightly back-left of B` | `A稍微在B后方`, `A稍微在B右后方`, `A稍微在B左后方` | `80` | `top_gap(A) <= top_gap(B) - 80` |
| default | `A is behind B`, `A is back-right of B`, `A is back-left of B` | `A在B后方`, `A在B右后方`, `A在B左后方` | `80` | `top_gap(A) <= top_gap(B) - 80` |
| clear | `A is clearly behind B`, `A is clearly back-right of B`, `A is clearly back-left of B` | `A明显在B后方`, `A明显在B右后方`, `A明显在B左后方` | `260` | `top_gap(A) <= top_gap(B) - 260` |
| far | `A is far behind B`, `A is far back-right of B`, `A is far back-left of B` | `A远在B后方`, `A远在B右后方`, `A远在B左后方` | `340` | `top_gap(A) <= top_gap(B) - 340` |
| extra-far | `A is very far behind B`, `A is extra-far back-right of B`, `A is extra-far back-left of B` | `A非常远在B后方`, `A非常远在B右后方`, `A非常远在B左后方` | `420` | `top_gap(A) <= top_gap(B) - 420` |

如果 `top_gap(B) - delta_px` 小于安全留白，目标值应被夹到安全范围内：

```text
target_top_gap(A) = max(min_visible_top_gap_px, top_gap(B) - delta_px)
```

建议默认：

```text
min_visible_top_gap_px = 80
```

该屏幕空间分级约束必须由 harness 的投影函数和数值移动函数执行，不允许依赖 LLM 根据 prompt 自行猜测 A 应该移动多少。

如果 screen-depth 分级导致前景物体因透视看起来过大，或背景物体看起来过小，不能取消前后关系。应交给 `LAYOUT_READABLE_DIMENSION_RULES.md` 中的 screen-size 工具处理：先尝试不破坏 `bottom_gap/top_gap` 分级的 `x` 调整，再尝试相机参数，必要时对目标 cube 做 `_screen_size_dim_compensated` 等比例尺寸补偿。最终 validation 必须同时满足空间谓词、screen-depth 分级和该对象对应的 screen-size ratio 范围。

## Screen-Space Left/Right Gap Levels

世界坐标的 `left_of/right_of` 只能保证左右边界顺序，不能保证投影后不互相贴近或遮挡。因此所有 `left_of/right_of` predicate 还必须执行屏幕空间左右硬间距。

```text
left_of(A, B):  bbox_right(A) <= bbox_left(B)  - gap_px
right_of(A, B): bbox_left(A)  >= bbox_right(B) + gap_px
```

强度分级和 front/back 使用同一组程度词，但查表为左右像素距离：

| Prompt strength | English examples | Chinese examples | gap_px |
| --- | --- | --- | ---: |
| slight | `slightly left of B`, `slightly front-right of B` | `稍微在B左侧`, `稍微在B右前方` | `24` |
| default | `A is left of B`, `A is right of B`, `A is front-right of B` | `A在B左侧`, `A在B右侧`, `A在B右前方` | `48` |
| clear | `A is clearly left of B`, `A is clearly front-right of B` | `A明显在B左侧`, `A明显在B右前方` | `80` |
| far | `A is far left of B`, `A is far front-right of B` | `A远在B左侧`, `A远在B右前方` | `120` |
| extra-far | `A is very far left of B`, `A is extra-far front-right of B` | `A非常远在B左侧`, `A非常远在B右前方` | `160` |

对于 `front-right/front-left/back-right/back-left`，harness 必须同时执行：

```text
1. 世界空间 front/back predicate。
2. 世界空间 left/right predicate。
3. screen-depth front/back 分级。
4. screen-lateral left/right 分级。
5. pairwise screen occlusion 上限。
```

这些规则由 `harness_param_constraints.py` 中的 parser、投影函数和数值搜索函数强制执行，不依赖 LLM 根据 prompt 猜测间距。

## Camera View Angle Notes

当前 harness 不从 prompt 中强制解析相机俯仰角。`agent_text2pkl_v5.py` 仍按规划器返回值初始化相机，默认示例值为 `camera_elevation_deg = 12.0`；后续验证失败时，确定性 repair 可以为了完整可见、关系正确和画面紧凑调整 `camera_data.camera_elevation`。

相机俯仰角在 pkl 中以 radian 保存。常规初始 fit 搜索使用 5-25 度附近的候选值，component compactness repair 可以通过 `camera_data.camera_elevation` 的普通 `set_param` 动作继续调整视角。

## Pairwise Screen Occlusion Limit

任意两个 cube 的投影 bbox 遮挡比例不得超过 `20%`：

```text
overlap_ratio = intersection_area(bbox(A), bbox(B)) / min(area(A), area(B))
overlap_ratio <= 0.20
```

如果超过上限，harness 优先沿已有 `left_of/right_of` 方向分离对象；没有显式左右关系时，按当前屏幕左右顺序选择分离方向。修复后仍必须重新验证空间谓词、screen-depth、screen-lateral、screen-size、full-visible 和 component-compactness。

连接写法也允许：

```text
A is behind and to the left of B -> behind(A, B) + left_of(A, B)
A is behind and to the right of B -> behind(A, B) + right_of(A, B)
A is in front of and to the left of B -> in_front_of(A, B) + left_of(A, B)
A is in front of and to the right of B -> in_front_of(A, B) + right_of(A, B)
```

## Explicit Non-Rules

以下短语不能解析为 support：

```text
on the right side of
on the left side of
right side of
left side of
在右边
在左边
在右侧
在左侧
```

例如：

```text
cat is lying on the right side of the sofa -> right_of(cat, sofa)
dog is lying on the left side of the cat -> left_of(dog, cat)
```

只有 `on top of`, `placed on`, `resting on`, `on the countertop of`, `在...上`, `放在...上`, `在...台面上` 这类明确支撑短语才会生成 `support`。

## Camera-Facing Azimuth Rules

对象朝向镜头的短语不生成空间 predicate，而是生成 `camera_facing` 约束，并直接设置 pkl 的 `subjects_data[*].azimuth[0]`。prompt 没有写朝向时也会生成默认约束。

默认规则：

```text
front/default -> azimuth = asset_default_azimuths[type]
back          -> azimuth = asset_default_azimuths[type] + pi
left side     -> azimuth = asset_default_azimuths[type] + pi/2
right side    -> azimuth = asset_default_azimuths[type] - pi/2
```

当前 `asset_default_azimuths[type] = 1.570796`，所以：

| Prompt 语义 | English examples | Chinese examples | azimuth rad |
| --- | --- | --- | --- |
| 正面朝向镜头 | `A faces the camera`, `A front faces the camera`, `the front of A faces the camera` | `A正面朝向镜头`, `A前面面向摄像机` | `1.570796` |
| 背面朝向镜头 | `the back of A faces the camera`, `A back faces the camera`, `A rear faces the viewer` | `A背面朝向镜头`, `A后面面向摄像机` | `4.712389` |
| 左侧朝向镜头 | `the left side of A faces the camera`, `A left side faces the viewer` | `A左侧朝向镜头`, `A左面面向摄像机` | `3.141593` |
| 右侧朝向镜头 | `the right side of A faces the camera`, `A right side faces the viewer` | `A右侧朝向镜头`, `A右面面向摄像机` | `0.000000` |

harness 只按这些短语和默认值强制写 `azimuth`。不要依赖 LLM 根据“看起来正面/侧面”自行猜角度。
