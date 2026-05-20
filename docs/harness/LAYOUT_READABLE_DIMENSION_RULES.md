# Layout-Readable Cube Dimension Rules

本文档定义 `inference/asset_dimensions.json` 的新目标：所有 cube 尺寸都服务于 harness 的空间布局、可见性和关系验证，不再追求真实物理尺寸或原始 mesh bbox。后续批量修改 86 类物体尺寸时，应以本文档为唯一规格来源。

同一套 cube 规范还包含 `inference/asset_default_azimuths.json`。尺寸表只定义 `[width, depth, height]`，默认朝向表定义每个 asset type 在“正面朝向镜头”时必须写入 pkl 的 `azimuth` 弧度值。harness 必须像投影默认尺寸一样投影默认朝向，不能让大模型自由猜测角度。

## 目标

`asset_dimensions.json` 中每个物体的 `[width, depth, height]` 应满足：

- 能让 3 到 8 个常见物体在 1024x1024 视图中完整可见。
- 能稳定表达 `left/right/front/behind/center/support/above/below` 等关系。
- 能保持物体类别之间的相对大小常识。
- 能让功能关系成立，例如小物体可放在 table/cabinet/desk 上，cat/dog 可放在 sofa/bed 上。
- 能减少 repair loop 的负担，使第一次 text-to-pkl 就尽量生成可通过 harness 的布局。

这里的尺寸不是米制真实尺寸。它是用于布局求解的规范化 cube 尺寸。

## 坐标和维度约定

每个尺寸数组固定为：

```text
[width, depth, height]
```

- `width` 控制左右方向占用，对应 scene `y` 方向。
- `depth` 控制前后方向占用，对应 scene `x` 方向。
- `height` 控制竖直方向占用，对应 scene `z` 方向。
- 所有值必须为正数，推荐保留 3 到 4 位小数。
- 常规物体的最大单轴尺寸不应超过 `2.6`，除非是 bus、helicopter、bulldozer 这类大物体。
- 小物体的最小可见尺寸不应低于 `0.08`，否则 cube 太小，投影和 mask 检查不稳定。

## 默认朝向规则

`inference/asset_default_azimuths.json` 的键必须与 `inference/asset_dimensions.json` 的 asset type 一一对应。每个值是弧度制 `azimuth`，表示该类型 cube 的正面朝向摄像机时的默认 yaw。

当前统一默认值：

```text
front face toward camera -> azimuth = 1.570796 rad
```

这与现有 cube 几何约定一致：在 pkl / renderer 中 cube 的局部 `width` 原本沿 `x` 轴，`depth` 原本沿 `y` 轴；设置 `azimuth = pi/2` 后，布局语义中的 `width` 对应画面左右 `y` 方向，`depth` 对应前后 `x` 方向。`sofa` 这类宽物体因此会正面朝向镜头，而不是只露出侧面。

harness 执行规则：

- prompt 没有明确朝向时，所有对象使用其 asset type 的默认正面朝向。
- `agent_text2pkl_v5.py` 生成初始 scene 后，`harness_param_constraints.py` 必须调用工具函数把 `subjects_data[*].azimuth[0]` 投影到默认值。
- `agent_opinion.py` / repair plan 也必须验证并修复 `camera_facing_azimuth`，不能只给自然语言建议。
- 修改 `asset_dimensions.json` 新增或删除类型时，必须同步更新 `asset_default_azimuths.json`。

### Prompt 朝向到 azimuth 的机械映射

朝向短语只决定“物体哪一面朝向镜头”，再由工具按默认正面角度加固定 offset。

```text
base = asset_default_azimuths[type]
front face toward camera -> azimuth = base
left side toward camera  -> azimuth = base + pi/2
right side toward camera -> azimuth = base - pi/2
back face toward camera  -> azimuth = base + pi
```

所有角度最终归一化到 `[0, 2*pi)`，保留 6 位小数。以当前默认 `base = 1.570796` 为例：

| Prompt 语义 | azimuth rad |
| --- | --- |
| 默认 / 正面朝向镜头 / front face toward camera | `1.570796` |
| 背面朝向镜头 / back face toward camera | `4.712389` |
| 左侧朝向镜头 / left side toward camera | `3.141593` |
| 右侧朝向镜头 / right side toward camera | `0.000000` |

这些值是 harness 的硬规则。LLM 可以输出任意初始 `azimuth`，但最终 pkl 必须由工具函数修正为上表结果。

## 总体尺寸区间

建议使用以下布局尺度带：

| Band | 用途 | 典型范围 |
| --- | --- | --- |
| tiny | 杯子、碗、手机、鞋、鸟、小玩具 | `0.08` 到 `0.35` |
| small | 猫、狗、小家电、灯、垃圾桶、背包 | `0.25` 到 `0.90` |
| medium | 椅子、桌子、柜子、人、沙发、床、家电 | `0.70` 到 `2.20` |
| large | 汽车、大动物、大家具 | `1.20` 到 `2.80` |
| xlarge | bus、helicopter、bulldozer 等 | `2.00` 到 `3.80` |

尺度带只规定相对量级，不要求每个轴都落在同一范围。例如 lamp 可以很窄但很高。

## 相对尺度规则

相对物体尺寸的主判断依据是 cube 体积：

```text
cube_volume = width * depth * height
```

讨论“物体 A 比物体 B 大/小”时，默认比较 `cube_volume`，而不是单独比较某一条边。单轴尺寸只用于形状语义和功能约束，例如 refrigerator 高、sofa 宽、vehicle 长。体积排序必须优先满足类别常识；形状比例只能在不破坏体积排序的前提下调整。

以下大小关系必须稳定成立：

- `bus > van/suv/pickup truck > sedan/coupe/sports car > motorbike/bicycle/scooter`
- `bed >= sofa > desk/table > chair/office chair/stool`
- `wardrobe/bookshelf/refrigerator > cabinet/drawer/washing machine/oven`
- `elephant > giraffe/horse/cow > deer/goat/sheep/pig > dog/fox/wolf > cat/rabbit`
- `man` 高度应明显大于 dog/cat，接近 chair/desk 的生活尺度。
- `table/desk/cabinet` 的顶面面积必须能容纳 cup/bowl/plate/book/phone/keyboard/microwave 等小物体。
- `sofa/bed` 的顶面面积必须能容纳 cat/dog/pillow/blanket/teddy。
- `tiny` 物体不能大到接近 chair/table。
- 同一语义组内部差异应小于跨组差异，例如 sedan/coupe/ferrari/bugatti/mclaren/lamborghini/vw beetle 的 cube 体积应接近。

### 体积带参考

以下体积带用于审核 `asset_dimensions.json`，不是绝对物理单位：

| Volume Band | cube_volume 范围 | 典型对象 |
| --- | --- | --- |
| micro | `0.0005` 到 `0.008` | mouse, phone, cup, bowl, plate, sparrow |
| tiny | `0.008` 到 `0.04` | small birds, vase, speaker, teddy, rabbit, pillow |
| small | `0.04` 到 `0.18` | cat, small appliances, plant, backpack, lamp, dog |
| lower-medium | `0.18` 到 `0.55` | chair, stool, toilet, cabinet, sheep/goat/pig |
| medium | `0.55` 到 `1.30` | table, desk, man, bookshelf, refrigerator, sedan, large animals |
| large | `1.30` 到 `3.20` | sofa, bed, suv, van, elephant, tractor |
| xlarge | `3.20` 以上 | bus, helicopter, very large vehicles |

审核时应打印完整体积排序。如果某个对象落入明显错误的体积带，优先调整尺寸表，而不是依赖 `enforce_volume_order` 在运行时补救。

### 体积排序验收

批量修改后至少检查以下体积不等式：

- `volume(bus) > volume(van) > volume(suv) > volume(sedan)`
- `volume(sedan) > volume(motorbike) > volume(bicycle) >= volume(scooter)`
- `volume(bed) > volume(sofa) > volume(table) > volume(chair) > volume(stool)`
- `volume(refrigerator) > volume(washing machine) >= volume(oven) > volume(microwave)`
- `volume(wardrobe) >= volume(refrigerator) > volume(bookshelf) > volume(cabinet) > volume(drawer)`
- `volume(elephant) > volume(giraffe) > volume(horse) >= volume(cow) > volume(deer) > volume(sheep) >= volume(goat) > volume(dog) > volume(cat)`
- `volume(man) > volume(office chair) >= volume(chair) > volume(stool)`
- `volume(man) > volume(dog)` and `volume(man) < volume(sofa)`
- `volume(table) > volume(microwave) > volume(bowl) >= volume(cup)`
- `volume(backpack) > volume(bottle) > volume(cup)`
- `volume(teddy) > volume(mouse)` and `volume(teddy) < volume(cat)`

这些不等式是 harness 尺寸表的最低体积一致性标准。若为了可见性或支撑关系需要打破某条不等式，必须在修改记录中说明原因。

### 体积与功能约束的优先级

体积排序不能破坏功能关系。若两个目标冲突，优先级如下：

1. 支撑关系可行：可放置物体的 footprint 必须小于支撑面。
2. 体积排序合理：同类和跨类对象的 `cube_volume` 顺序应符合常识。
3. 形状语义合理：高、宽、长、薄等单轴特征要可辨认。
4. 图像可见性稳定：常见场景中 cube 不出框、不小到不可见。

例如 dog 的体积必须大于 cat，但 dog 的 `width/depth` 仍应小到能放在 sofa/bed 上；microwave 的体积应大于 bowl/cup，但 footprint 仍应能放在 cabinet/table 上。

## 功能关系规则

尺寸必须让常见 support 关系几何可行。

支撑面物体：

- `table`, `desk`, `cabinet`, `drawer`, `bookshelf`, `bed`, `sofa`, `refrigerator`, `oven`, `microwave`
- 这些物体的 `width` 和 `depth` 应大于可放置物体的对应轴，并留出至少 `0.02` 到 `0.08` 的 inset。

可放置小物体：

- `cup`, `bowl`, `plate`, `bottle`, `phone`, `keyboard`, `clock`, `book/books`, `vase`, `plant`, `speaker`, `microwave`
- 这类物体的 footprint 应显著小于 table/desk/cabinet 顶面。

可放在 sofa/bed 上的物体：

- `cat`, `dog`, `pillow`, `blanket`, `teddy`
- `cat/dog` 的 footprint 应能放入 sofa/bed 顶面，但不要小到不可见。

不应默认作为支撑面的物体：

- 动物、车辆、人、门、窗、帘子、风扇、挂架等。
- 这些物体即使 geometrically large，也不应因为尺寸大而被 harness 当成常规 support surface。

## 屏幕尺寸合理性约束

`asset_dimensions.json` 定义的是布局参考尺寸，但最终图像还存在透视远近：同一尺寸的前景 cube 会显得更大，背景 cube 会显得更小。因此 harness 必须增加一层投影后的 screen-size 校验，不能只依赖 LLM prompt 或世界坐标尺寸。

对每个 cube 投影得到屏幕 bbox：

```text
bbox_width  = projected_bbox.max_x - projected_bbox.min_x
bbox_height = projected_bbox.max_y - projected_bbox.min_y
bbox_area   = bbox_width * bbox_height
```

再用 `asset_dimensions.json` 中同类型参考尺寸计算密度：

```text
height_density = bbox_height / asset_height
area_density   = sqrt(bbox_area / (asset_width * asset_height))
density        = 0.75 * height_density + 0.25 * area_density
```

一个 scene 内所有 cube 的 `density` 取中位数作为 `target_density`。每个 cube 必须满足：

```text
screen_size_ratio = density / target_density
```

允许范围由 prompt 中的前后关系机械决定，不能让 LLM 自行猜测：

| 对象关系 | screen_size_ratio 允许范围 | 目的 |
| --- | --- | --- |
| 普通物体，无 front/back 谓词 | `[0.8, 1.2]` | 保持常规屏幕尺寸一致性 |
| `front` / `front-left` / `front-right` | `[0.8, 1.6]` | 前景物体允许因透视显得更大 |
| `clearly front` | `[0.75, 1.8]` | 明显前景允许更大的屏幕占比 |
| `far front` / `very far front` / `extra-far front` | `[0.7, 2.0]` | 强前景允许更大的透视放大 |
| `back` / `back-left` / `back-right` / `behind` | `[0.55, 1.2]` | 背景物体允许因透视显得更小 |
| `clearly back` / `clearly behind` | `[0.45, 1.15]` | 明显背景允许更小的屏幕占比 |
| `far back` / `very far back` / `extra-far back` | `[0.35, 1.1]` | 强背景允许更大的透视缩小 |

无论 screen-size 如何补偿，`dims` 都必须保持在 `asset_dimensions.json` 默认尺寸的 `[0.7x, 1.4x]` 范围内。该边界是 per-asset 物理尺寸保护：例如 cat/dog 即使触发 screen-size compensation，也不能被缩到低于默认尺寸的 70%，不能被放大到高于默认尺寸的 140%。

执行优先级必须固定为工具/函数逻辑：

1. 先执行 `asset_dimensions.json` 默认尺寸、空间谓词、默认朝向、组件居中和 screen-depth 前后分级。
2. 对 screen-size 过大的 cube，优先沿 `x` 远离相机；对 screen-size 过小的 cube，优先沿 `x` 靠近相机。
3. 如果该 cube 同时有 `front/behind` screen-depth 约束，`x` 调整不能破坏 `bottom_gap/top_gap` 分级。
4. 如果单独调 `x` 不能把比例带回该对象的允许范围，再尝试 `camera.lens` 和 `camera.global_scale`，但不能让任何 cube 出画，也不能让已有 screen-depth 更差。
5. 如果前景对象仍过大，且继续缩小会触碰 `[0.7x, 1.4x]` 的下限，则可把不参与空间谓词边界的中性参考物体在 1.4x 上限内等比例放大，提高 scene median density，而不是继续缩小前景物体；被放大的参考物体必须写入 `_screen_size_dim_compensated = true`。
6. 仍不能满足时，才允许对目标 cube 做等比例 `dims` 补偿，并写入 `_screen_size_dim_compensated = true`；补偿结果仍必须被 `[0.7x, 1.4x]` 默认尺寸边界 clamp。
7. 如果对象已经到达 0.7x 下限但 screen-size 仍略高于 front 上限，物理尺寸下限优先，validation 不应继续要求突破下限缩小；如果对象已经到达 1.4x 上限但 screen-size 仍略低于 back 下限，同理以上限优先。
8. 无 front/back screen-depth 的 left/right 普通关系对象不能被 screen-size 工具单独沿 `x` 反复拉动，因为这会和 `left_of/right_of` 的 `x` 对齐规则震荡；这类对象若 screen-size 仍偏小/偏大，只能通过相机或 `[0.7x, 1.4x]` 范围内的 dims 补偿处理。
9. screen-depth 的像素验证使用 `3px` 容差，避免 1-2px 的投影误差造成 repair loop 震荡。
10. 补偿后必须再恢复 screen-depth，使前/后关系和 screen-size 同时通过 validation。

带 `_screen_size_dim_compensated` 的 cube 表示其 `dims` 是为了抵消透视导致的屏幕尺寸偏差而由 harness 机械生成。此时 `dimension_range` 和 `volume_order` 不应再把它强行拉回原始参考尺寸，否则会和 screen-size 约束循环冲突。

## 最终组件构图紧凑约束

在所有 cube 的世界空间关系、screen-depth、screen-size 和 full-visible 修复之后，harness 必须执行最终相机紧凑约束。目标是让“所有 cube 的集合”作为画面主体，占据足够大的屏幕范围，避免 layout 正确但主体过小、四周留白过多。

该约束只使用屏幕空间，不依赖大模型 prompt。对所有 cube 的投影 bbox 取 union bbox：

```text
component_width_ratio  = union_bbox_width  / image_width
component_height_ratio = union_bbox_height / image_height
component_area_ratio   = union_bbox_area   / (image_width * image_height)
```

验收以 `component_width_ratio` 和 `component_height_ratio` 为主，不单独依赖面积。横向展开场景的 union bbox 中间可能有大量空白，单看面积会误判；宽度和高度必须分开控制。

默认最低比例按场景类型机械选择：

| 场景类型 | component_width_ratio 最低值 | component_height_ratio 最低值 |
| --- | ---: | ---: |
| 单物体 | `0.35` | `0.45` |
| 2-3 个物体，普通关系 | `0.55` | `0.38` |
| 通用 3-6 个物体 fallback | `0.65` | `0.32` |
| 4-6 个物体，横向展开 left/right 明显 | `0.70` | `0.30` |
| 4-6 个物体，有 front/back 层次 | `0.65` | `0.34` |
| 7 个以上物体 | `0.78` | `0.28` |
| 明显纵向物体为主，例如 bookshelf/lamp/tower | `0.45` | `0.50` |

如果一个场景同时符合多类条件，取更严格的最低值。例如 5 个物体同时包含 `left/right` 和 `front-left/front-right` 时，目标为：

```text
component_width_ratio  >= max(0.70, 0.65) = 0.70
component_height_ratio >= max(0.30, 0.34) = 0.34
```

该规则不设置显式上限。执行策略是：先满足最低占比，然后在不破坏其他规则的前提下继续增大主体，直到安全边距或其他硬约束阻止继续放大。上限由以下条件自然决定：

```text
1. 所有 cube 必须完整可见。
2. union bbox 外侧必须保留安全边距。
3. spatial predicates 不能被破坏。
4. screen-depth relation 不能最终失败。
5. screen-size reasonableness 不能最终失败。
6. camera_data 必须保持在 lens/global_scale/elevation 合法范围内。
```

安全边距默认：

```text
COMPONENT_COMPACT_MARGIN_PX = 32
COMPONENT_COMPACT_MIN_MARGIN_PX = 20
```

物体很多或横向特别宽时，允许使用 `20px`，但不得低于该值。

相机修复优先级：

```text
1. 优先增大 camera.lens。
2. 必要时小幅增大 camera.global_scale。
3. 如果只调 lens/global_scale 会触发出画、遮挡或 screen-size 失败，允许联合搜索 camera_elevation。
4. 不优先移动 cube。
5. 不优先修改 dims。
6. 相机收紧后必须再次执行 screen-depth、screen-lateral、pairwise occlusion 和 screen-size 验证/修复。
```

这条规则的意义是让 camera framing 由 harness 函数机械决定，而不是让 LLM 在 prompt 中猜“更近一些”“占画面大一些”。

最终相机候选必须满足：

```text
1. 所有 cube 完整可见。
2. union bbox 安全边距满足目标。
3. spatial predicates 不失败。
4. screen-depth 前后分级不失败。
5. screen-lateral 左右硬间距不失败。
6. 任意两个 cube 的 bbox 遮挡比例不超过 20%。
7. screen-size reasonableness 不失败。
8. camera_data 保持在合法范围内。
```

## 类别规则

### 室内家具

适用对象：

```text
bed, bookshelf, cabinet, chair, curtain, desk, door, drawer, office chair,
sofa, stool, table, wardrobe, window
```

规则：

- `bed`: 宽深都大，height 中等，可承载 pillow/blanket/cat/dog。
- `sofa`: width 明显大于 depth，height 中等，可承载 cat/dog/pillow/teddy。
- `table/desk`: width/depth 中等，height 低于 man，顶面可放小物体。
- `chair/office chair/stool`: footprint 小于 table，height 不应超过 wardrobe/bookshelf。
- `bookshelf/wardrobe/door/window/curtain`: height 高，depth 浅。
- `cabinet/drawer`: 中等宽度，中等高度，depth 不应过大。

### 家电和电子设备

适用对象：

```text
air conditioner, computer, fan, keyboard, lamp, microwave, oven, phone,
refrigerator, sink, speaker, television, toilet, trash can, washing machine
```

规则：

- `refrigerator`, `wardrobe`, `door` 是高物体，height 应接近或高于 man。
- `microwave`, `computer`, `keyboard`, `phone`, `speaker` 是可放置物体，footprint 应可放在 desk/table/cabinet 上。
- `television` 宽大、depth 浅、height 中等。
- `lamp` footprint 小、height 高，不能占用过大横向空间。
- `washing machine`, `oven`, `sink`, `toilet` 是中等体块，不能接近 refrigerator 的高度。

### 小型生活物体

适用对象：

```text
backpack, blanket, bottle, bowl, clock, cup, hanger, mouse, pillow, plant,
plate, shoe, suitcase, teddy, vase
```

规则：

- `cup/bowl/plate/bottle/phone/mouse` 必须是 tiny 或 small，能放上 table/cabinet。
- `blanket/pillow` 可以较扁，height 应明显小于 width/depth。
- `plant/vase` footprint 小，height 可以较高，但不能接近 lamp/refrigerator。
- `suitcase/backpack` 是可见小物体，体积大于 cup/bowl，小于 chair/table。
- `teddy` 应能放在 sofa/bed/table 上，不能接近成人或椅子大小。

### 人和动物

适用对象：

```text
bear, cat, cow, crow, deer, dog, elephant, fox, giraffe, goat, hen, horse,
kangaroo, lion, man, pig, pigeon, rabbit, sheep, sparrow, tiger, wolf, zebra
```

规则：

- 四足动物通常 `depth > width`，height 与类别大小匹配。
- 鸟类 footprint 小，height 小到中等，但不能低于最小可见尺寸。
- `cat/rabbit` 应明显小于 dog。
- `dog/fox/wolf` 应小于 goat/sheep/pig。
- `horse/cow/zebra` 应大于 deer/goat/sheep/pig。
- `elephant/giraffe` 是最大动物组，不能和 dog/cat 同量级。
- `man` height 约为 medium 高物体，footprint 小于 sofa/bed/table。
- 动物不能为了真实 mesh bbox 保留异常大的 depth，例如接近 `2.0` 的 cat/dog depth 不适合作为布局尺寸。

### 车辆和交通工具

适用对象：

```text
bicycle, bugatti, bulldozer, bus, coupe, ferrari, helicopter, jeep,
lamborghini, mclaren, motorbike, pickup truck, scooter, sedan, suv, tractor,
van, vw beetle
```

规则：

- 车辆通常 `depth > width > height`。
- 同类跑车和轿车尺寸应接近，不应因为 mesh bbox 产生大幅差异。
- `bus` 是最大陆地车辆，明显大于 van/suv。
- `pickup truck/van/suv/jeep` 应大于 sedan/coupe。
- `bicycle/motorbike/scooter` footprint 小于汽车，height 可接近 man 腰肩高度。
- `helicopter` footprint 可大，但 height 不应过高到破坏视图。
- 大型车辆场景应降低同场景小物体数量，或允许 camera fit 调整。

## 形状比例规则

尺寸不仅要有总大小，还要符合形状语义：

- 扁平物体：`blanket`, `plate`, `keyboard`, `phone`, `window`, `curtain`, `door` 的某一轴应明显较薄。
- 高窄物体：`lamp`, `vase`, `bottle`, `refrigerator`, `wardrobe`, `door` 的 height 应是主要轴。
- 长物体：`bed`, `sofa`, `vehicle`, `animal` 的 depth 或 width 应明显长于另一横向轴。
- 方体物体：`cabinet`, `drawer`, `washing machine`, `oven`, `trash can` 可接近规则盒体。
- 支撑面物体：`table`, `desk`, `bed`, `sofa` 的 top footprint 必须足够大。

## 可见性规则

每个尺寸修改后，必须通过以下可见性原则：

- 单个物体不应填满画面。
- 典型 5 物体场景应能通过 `E_full_cube_visible`。
- tiny 物体在 1024x1024 下仍应有可辨识 cube，最小边建议不低于 `0.08`。
- 大物体和小物体同场景时，小物体不能小到被大物体完全遮挡。
- 大组件居中应由整体平移完成，尺寸表不应通过把物体缩到不合理来解决出框。

## 与 VISUAL_TARGET_DIMS 的关系

如果 `asset_dimensions.json` 全部改成布局可读尺寸，则 `VISUAL_TARGET_DIMS` 应逐步退化为临时兼容层：

- 新尺寸表覆盖全部 86 类物体后，`VISUAL_TARGET_DIMS` 不应再保存常规物体尺寸。
- 只允许保留少量特殊 override，例如某个 benchmark 临时需要更小的 `cup`。
- harness 的尺寸目标应优先来自 `asset_dimensions.json`，避免两套尺寸标准长期并存。

同时需要注意 `object_scales.py`：

- 如果 `asset_dimensions.json` 已经是最终布局可读尺寸，后续尺寸计算不应再乘原来的物体缩放系数后当作 harness 目标尺寸。
- 推荐在 harness 路径中直接使用 `default_dims_map`，不要用 `reference_dims_map = asset_dimensions * object_scales * DEFAULT_REFERENCE_DIM_SCALE` 作为最终尺寸目标。
- 如果仍保留 `object_scales.py` 给生成图像 prompt 或旧 pipeline 使用，必须在文档中明确它不参与 harness 尺寸归一化。

## 86 类物体覆盖清单

修改 `asset_dimensions.json` 时必须覆盖以下全部类型：

```text
air conditioner, backpack, bathtub, bear, bed, bicycle, blanket, bookshelf,
bottle, bowl, bugatti, bulldozer, bus, cabinet, cat, chair, clock, computer,
coupe, cow, crow, cup, curtain, deer, desk, dog, door, drawer, elephant, fan,
ferrari, fox, giraffe, goat, hanger, helicopter, hen, horse, jeep, kangaroo,
keyboard, lamborghini, lamp, lion, man, mclaren, microwave, motorbike, mouse,
office chair, oven, phone, pickup truck, pig, pigeon, pillow, plant, plate,
rabbit, refrigerator, scooter, sedan, sheep, shoe, sink, sofa, sparrow,
speaker, stool, suitcase, suv, table, teddy, television, tiger, toilet,
tractor, trash can, van, vase, vw beetle, wardrobe, washing machine, window,
wolf, zebra
```

任何新增类型也必须先归入一个功能类别和尺度带，再写入尺寸。

## 修改流程

每次批量修改尺寸，应按以下步骤执行：

1. 先按类别分组，不要逐个物体凭感觉改。
2. 为每组确定基准物体，例如 `sedan`, `man`, `table`, `sofa`, `cat`, `cup`。
3. 先定基准物体尺寸，再按相对尺度派生同组物体。
4. 计算全部对象的 `cube_volume = width * depth * height` 并打印排序。
5. 检查本文档的体积排序验收不等式。
6. 对 support surface 检查 footprint，确保可放置物体能落入支撑面。
7. 对 tiny object 检查最小可见尺寸，避免低于 `0.08`。
8. 对 large/xlarge object 检查单体出框风险，避免常规 prompt 中撑满画面。
9. 运行 golden prompts，记录通过率和失败类型。
10. 只在失败类型明确指向尺寸问题时调整尺寸，不用尺寸掩盖关系解析错误。

## Golden Prompt 验收集

尺寸表提交前，至少覆盖以下场景：

- kitchen countertop: refrigerator, cabinet, microwave, bowl, cup
- living room: sofa, cat, dog, lamp, table
- bedroom: bed, pillow, blanket, teddy, lamp
- office: desk, office chair, computer, keyboard, phone
- dining/tabletop: table, bowl, plate, bottle, vase
- vehicle scene: sedan, bicycle, man, traffic-sized object
- animal scale scene: elephant, horse, dog, cat
- mixed small/large: refrigerator, cup, bowl, plant

每个场景至少检查：

- `A_type_count_match`
- `B_relation_match`
- `C_physical_plausibility`
- `D_size_reasonable`
- `E_full_cube_visible`
- harness `dimension_ranges`
- harness `spatial_predicates`

## 验收标准

一次尺寸表修改可以接受的最低标准：

- 86 类物体都有明确尺寸，无遗漏。
- 所有尺寸均为正数，格式为 `[width, depth, height]`。
- 无常见小物体任意一轴低于 `0.08`，除非有专门理由。
- 无普通室内物体任意一轴超过 `2.6`，除非属于大交通工具或大型动物。
- golden prompts 中尺寸相关失败数量显著下降。
- 不引入新的 support fit 失败。
- 不通过缩小所有物体来规避可见性问题。

## 不应做的事

- 不要把真实世界米制尺寸直接写进 `asset_dimensions.json`。
- 不要直接使用 raw mesh bbox，尤其是动物和车辆类。
- 不要让 `VISUAL_TARGET_DIMS` 和 `asset_dimensions.json` 长期保存两套互相冲突的尺寸。
- 不要为了让一个 prompt 过，把全局类别尺寸改到破坏其他类别关系。
- 不要用尺寸修改修复文本关系解析错误，例如把 `right side of` 错当 support 的问题必须在 predicate 解析层解决。
