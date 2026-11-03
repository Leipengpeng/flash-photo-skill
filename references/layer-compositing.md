# V2：只生成光照层，合成回原图

用户选择“模型直接生成光照蒙版”时使用 V2。深度、材质与逐灯规划沿用原有闭环，图像模型只提供较低分辨率的光照控制图，原始照片始终由本地合成器读取。用户明确要求先生成完整打光结果、相减并滤波时，转到 [V3 Delta 迁移实验](delta-transfer.md)，不要用本节的直接蒙版约束阻止该策略。完整结果生成模板见 [V1 提示词](maps-and-prompts.md#v1-整图编辑模板)。

## 原图与控制图各自负责什么

| 文件 | 用途 |
| --- | --- |
| 用户原始照片 | 人物、物件、文字、纹理和最终像素网格的来源；原件不覆盖 |
| 显示方向正确的缩小参考图 | 供模型看构图；全画幅缩小，不裁切，也不先美化 |
| `depth.png`、`materials.png` | 经审查的相对深度和材质参考；不当作标定数据 |
| `gain.png` | 中性灰为零的 RGB 增益场，控制漫反射补光的区域、方向与强弱；不包含照片底图 |
| 可选 `highlights.png`、`rim.png` | 黑色为零的加性光层，分别表达局部反光或轮廓新增光；与增益场的编码不同 |
| 可选 `coverage.png` | 单独核对的灰度覆盖范围：黑色保护原图，白色允许加光，灰色过渡 |
| 原尺寸合成图与独立效果层 | 确定性脚本的输出；合成后仍须实际审图 |

先记录原文件哈希、存储宽高、EXIF 方向与显示宽高。EXIF 旋转后的显示尺寸才是最终照片的目标网格；宽高互换不算裁切。控制图可以较小，但须保持全画幅、方向和宽高比。错位不能靠放大后模糊来掩盖，也不能扭曲原图去迁就蒙版。

本流程保留的是原始照片的尺寸、构图网格和细节来源。颜色会变化，亮部压缩、色域转换与输出量化也会影响像素值；不称为“原图每个像素不变”或“无损恢复高光”。

## 两种光照层的编码

控制图均为不透明的 8 位 RGB PNG，但用途与零点不同，不能混用。

**漫反射增益场：中性灰 128 为零。** 每个颜色通道分别编码相对曝光变化：

```text
EV[channel] = (code[channel] - 128) / 127 × 2
gain[channel] = 2 ** EV[channel]
```

- `RGB(128,128,128)` 表示零变化。
- 三通道一起升高表示中性加光；红通道相对更高可表示暖光，蓝通道相对更高可表示冷光。
- 默认 `add-only`：负 EV 截为零。阴影表示新增灯光被遮挡，该处保持原环境光，而非涂黑已有照片。
- 只有用户允许减光或改变环境曝光时，才使用有符号变化，并记录其范围和原因。

这个编码是合成器的文件约定。图像模型画出的灰度与颜色仍是生成式估计，不能据此报告实测 EV、灯距或功率。模型也不保证把未照区域精确画成 `128`；即使肉眼看着是灰底，偏亮几个码值也可能让整片背景被提亮。检查中性码值与彩色增量的范围，对已核对的天空、远景等区域使用独立覆盖蒙版保护。深度/材质类别色不能混进光照场。

**加性高光或轮廓层：黑色 0 为零。** 它表达在原像素之外增加的光，不再乘以原片肤色或衣料颜色：

```text
absolute_light[channel] = code[channel] / 255 × additive_scale
```

三通道相等可表达中性反光；不同通道可表达暖色或冷色新增光。`--additive-scale` 默认 `0.2`，即代码 `255` 对应高光压缩前 `0.2` 的绝对线性 sRGB 新增值。代码值按线性数据解释，不做 sRGB 解码或 ICC 转换；它不是显示颜色或实测辐射值。未照区域必须为黑，不能沿用增益场的 `128` 灰底，否则会增加整片雾状亮光。

初版单纯放大原像素时，皮肤容易成为均匀的暖粉色提亮，暗头发也难以获得亮色轮廓；把所有灯的增益都铺满脸，还会让预设看起来相同。需要白色集中反光或暗部轮廓时，单独生成加性层；不能只提高全脸增益。加性层仍须服从同一灯位、表面朝向与遮挡，不以白斑或彩色描边冒充真实反射。

原始照片通过嵌入 ICC 转换到 sRGB；没有 ICC 时按 sRGB 解释并记录。合成在线性 sRGB 中完成，合计正增量后用共同通道系数压缩亮部，避免逐灯叠加硬截断。强而接近中性的绝对高光进入亮部时，另用艺术化显示肩部让反光逐渐趋近中性白，减少肤色红通道把白色高光限制成橙色的问题；彩色轮廓、仅增益的光和未改像素不受这一步作用。这不是实测传感器溢出或 BRDF 估计。

最终输出为带 sRGB ICC 的 8 位 PNG，不复制原 EXIF/GPS。实际参数、色彩处理与输出位深写入报告，不把结果称为保留了原有广色域或 RAW 动态范围。

## 生成与核对光照层

按 [辅助图与提示词](maps-and-prompts.md) 先实际查看原图，核对或复用深度、材质图，再写逐灯简报。至少明确主体、近背景与远背景的响应，以及鼻影、下颌、发肩轮廓、镜片等本图实际存在的区域。没有相关内容就删去该要求。

模型输入的角色要写清楚：

1. 原图或其全画幅缩小参考：唯一构图和物体边界权威。
2. 深度图：只帮助判断远近与遮挡。
3. 材质图：只帮助判断表面响应，类别色不进入输出。

本地输入按工具支持的路径机制传入。若工具不能同时接收这些图，传原图并把已审的辅助图结论写入提示词，记录实际输入。不要虚构深度条件接口。

```text
Use case: lighting-weather.
Asset type: RGB lighting-gain control field for later compositing, NOT a photograph.
Image 1: the original full-frame photograph, sole geometry and silhouette authority.
Image 2: reviewed inferred relative-depth guide only.
Image 3: reviewed semantic material guide only; never copy its class colors.

Generate ONLY one flat, opaque RGB lighting-control image registered to image 1.
Use the same complete framing, orientation, aspect ratio and object boundaries.
Do not return a relit photo, a copy of the photo, a cutout, or a before/after sheet.
Do not put any original RGB texture, skin detail, hair strands, fabric pattern,
writing, scenery detail, labels, arrows or lighting hardware into the control field.
Use flat, faceless 2D mask regions, not a shaded portrait or a grayscale photograph.
No eyes, eyebrows, mouth, pupils, eyeglass lines, individual hair strands or cloth checks.
Represent local light accents as simple smooth intensity patches, not drawn anatomy.

Encoding: neutral RGB(128,128,128) means no added light.
Per channel, code = 128 + 127 * (requested relative EV / 2).
Use smooth coherent illumination regions with source-appropriate shadow edges.
Higher equal RGB values mean neutral added light; channel differences mean color.
Keep unlit and protected regions exactly neutral gray, not black or transparent.
This is an artist-directed relative-light estimate, not measured light transport.

Added sources and their separate roles: {light_sources and per-source responses}.
Foreground / subject / near background / far background: {region-specific plan}.
Occlusion and shadow receivers: {actual chains and appropriate penumbra}.
Specular regions and emitter shape where justified: {conditional paths}.
Protected regions that remain neutral: {sky, distant objects, other verified regions}.
Observed scene anchors and silhouettes: {at least three checked anchors}.

Return only the encoded RGB field. No photographic base, legend or decorative glow.
```

V2 地图中应当看得见灯光作用的区域，但不应当重新画出一张照片。灰度照片也不合格：眼睛、眉毛、嘴、瞳孔、眼镜线条、发丝或衣服格纹清楚可见时，仍会把生成结构带进原片。针对光照场重做一次，明确要求无五官、无纹理的平面强度区域。不能把完整照片求差冒称为“模型直接生成蒙版”；用户选择的 V3 则明确采用完整参考求差，另行记录并过滤其重绘漂移。

需要加性层时，保留相同输入角色与几何约束，将上面模板中的编码与输出要求替换为：

```text
Asset type: black-zero additive-light data layer, NOT a photograph or an EV map.
The entire unlit canvas must be RGB(0,0,0). No gray background.
Per-channel code / 255 denotes positive absolute linear-light addition,
scaled later by the compositor; it is not an sRGB display color.
Generate ONLY the compact reflection patches or finite rim-light regions
supported by the planned source and the original surface geometry: {regions}.
Keep eyes, pupils, mouth and unrelated regions black; do not draw anatomy,
hair strands, skin pores, fabric texture, scenic content, borders or hardware.
Use locally feathered reflection shapes, not white stickers, polka-dots,
a broad face flood or a continuous glowing outline around the whole person.
Return only the registered additive data layer.
```

分别核对两类层的背景码值、区域边界和锚点。鼻尖、颧区、镜片和发缘彼此邻近，生成高光落进镜片或背景时，不能靠降低整层强度掩盖错位。可基于原图锚点只修正控制层的位置与边缘，并保留原层、变换参数和审图记录；原片不随控制层移动。平滑只能抑制控制图中的杂纹，不能当作几何配准。

## 让光源在缩览中有区别

- **近轴直闪**：前向受光面明确变亮，皮肤上有位置合理的紧凑中性反光；鼻下、下颌或眼镜造成的新光遮挡短而有边界。同一灯的亮面、高光和遮挡方向须一致。不要把整张脸刷成均匀粉橙色，也不要为了硬影涂黑已有环境光。
- **侧面大柔光箱**：来光侧的脸与衣料受光明确，背光侧保留更多原环境；过渡宽，反光面积可更大且更柔。做明显侧光对照时可选更偏侧的角度，如画面左侧约 70°，但不把这个示例固定成所有柔光请求的灯位。
- **主灯加轮廓灯**：主灯保持不变，新增灯只在有条件受光的发缘和肩缘产生有限、可断开的亮边。轮廓位于真实边界内，检查暗发上的新增亮度与暖冷色；不能画成整圈霓虹线。

先看正常视频缩览，再看原像素局部。强闪应能从局部反光、亮面与遮挡读出“灯加上了”；只提高全脸平均亮度不能算达到。比较预设时，还要能在预定观察区指出方向、软硬或轮廓贡献的区别。用户说太弱时，可复用图层提高对应贡献或重做主要失败的控制层；需要强度梯度时再出弱、中、强对照，不要求日常请求固定多出图。实际审图前不宣称改善完成，用户是否认可仍由用户判断。

## 边缘、颜色和多灯

先在控制图尺度平滑不该出现的细碎纹理，再放大到原始网格。增益场使用原片引导边缘；加性层使用独立的小尺度平滑和双线性放大，避免紧凑高光被漫反射的大邻域滤波抹掉。原片纹理不做模糊。羽化宽度要符合光源：大柔光箱有较宽的过渡，直闪的短影不能被同一层大模糊抹掉。发丝、眼镜与肩缘要检查漏光、暗边、亮边和连续描线；边缘引导不能自动证明语义正确。

`--additive-texture` 默认 `0`。强加性光显得过平时，可在 `0..1` 内试用原片亮度的局部对比，对新增光做有界调制；它仅作用于新输入的加性图，不修改原片，也不生成细节。该启发式不是材质反射率或法线估计，可能同时强调原噪声、边缘和原有照明，须看局部效果；不能称为真实纹理恢复。保存的 NPZ 已包含调制结果，复用时不重复这一步。

覆盖蒙版与光照场分开。覆盖蒙版用于保护天空、远景和未受光区域；它不替代光照场内部的明暗与色彩变化。材质图是生成式分类参考，未经对齐核查不能直接当精确抠图蒙版。对硬保护区，合成后核查其像素是否保持不变。

双灯仍逐源规划。需要独立开关主灯和轮廓灯时，分别保存控制图。合成器支持重复 `--field` 与 `--additive-field`：增益场先把 EV 变为 `2 ** EV - 1` 的贡献，加性层直接提供绝对线性新增光；按各源权重合计，最后只做一次高光压缩。不能直接相加 EV，也不能把已经压缩的成图逐层叠灯。

```text
source_EV_i = filtered_EV_i × strength × coverage
combined_gain = max(1 + sum(field_strength_i × (2 ** source_EV_i - 1)), 1e-8)
absolute_light_j = filtered_code_j / 255 × additive_scale
added_radiance = sum(additive_strength_j × absolute_light_j × strength × coverage)
linear_target = original_linear_RGB × combined_gain + added_radiance
linear_result = one_highlight_rolloff(original_linear_RGB, linear_target)
```

公式略去可选的加性纹理调制，以及强中性高光的艺术化趋白显示处理；完整实际参数以脚本与报告为准。

这里的 `field_strength` 和 `additive_strength` 都是未标定的贡献权重，不是某款闪灯的实测功率。`--strength` 对增益场是 EV 倍率，对加性光是线性倍率；零值关闭两类贡献。优先用各层自己的权重调节，避免混淆。每张单灯图与总图都使用用户原图作底，不把一张低分辨率成图当作下一盏灯的输入。也可以生成一张已含两源的合并场，但合并后只能整体调节，不能声称已分离两盏灯。

原图接近全黑的区域缺少可用信号，增益不能恢复不存在的细节；加性光虽能把黑像素照亮，也不会恢复其原有纹理。过曝区域同样不能恢复已丢失的信息。新增亮光会改变局部颜色与细节对比，强高光还可能压低可见纹理，不能称为微纹理逐像素不变。低分辨率控制图对细窄反射、发丝和硬影的位置控制有限，不能靠生成新皮肤、毛发和文字补齐。

## 合成与交付

使用随技能提供的 `scripts/relight_layers.py`，依赖 Python、Pillow、NumPy、SciPy。脚本处理图层运算，不替代图像模型生成光照场。

```bash
python scripts/relight_layers.py prepare --source original.jpg --out work/reference --max-edge 1600
python scripts/relight_layers.py composite --source original.jpg --field gain.png --out work/flash --save-arrays
```

`prepare` 生成转正后的全画幅缩小参考 `prepared.png` 与 `prepare.json`。`composite` 始终重新读取 `--source` 原件，不从 `prepared.png` 合成大图。

直闪需要独立的集中反光时，加入已审过的黑底加性图；下面数值是示例，须按成图审查调整：

```bash
python scripts/relight_layers.py composite --source original.jpg --field diffuse.png --field-strength 1 --additive-field highlights.png --additive-strength 1 --additive-scale 0.2 --additive-smooth 0.8 --mask coverage.png --out work/flash --save-arrays
```

只检查高光或轮廓层时，可以省略 `--field`，只传 `--additive-field`。两类输入至少有一种。加性图按原图实际边界定位，天空和远景维持黑底，并用经审的覆盖蒙版保护。

主灯与轮廓灯分别生成后，可以调整各灯权重；下面主灯为 `1`、轮廓灯为 `0.6`：

```bash
python scripts/relight_layers.py composite --source original.jpg --field main.png --field-strength 1 --field rim.png --field-strength 0.6 --out work/two-lights --save-arrays
```

要只看主灯，把第二个 `--field-strength` 改为 `0`，其余输入不变。提供权重时数量必须与 `--field` 一一对应，顺序相同；全部省略时各灯权重为 `1`。

若主灯用增益场、轮廓用加性场，则分别调两类权重：

```bash
python scripts/relight_layers.py composite --source original.jpg --field main.png --field-strength 1 --additive-field rim.png --additive-strength 0.6 --additive-scale 0.2 --out work/key-and-rim --save-arrays
```

把 `--additive-strength` 设为 `0` 即关闭这份轮廓贡献。多份加性图的权重与 `--additive-field` 按顺序一一对应，全部省略时各为 `1`。一盏灯若同时有增益和加性高光，两份输入必须保持同一灯位；也可先分别保存整盏灯的 NPZ，再通过重复 `--field` 组合。

已经保存浮点场后，可以复用，不再调用图像模型：

```bash
python scripts/relight_layers.py composite --source original.jpg --field work/flash/relight_arrays.npz --field-strength 0.7 --out work/flash-weaker --save-arrays
```

NPZ 内的 `ev` 保存漫反射总增益，`additive_linear` 保存独立加性光，两者均在高光压缩之前，已包含原合成时的覆盖、强度与权重。用 `--field` 读取 NPZ 时，合成器同时复用两部分；旧 NPZ 没有加性数据时按零处理，不必额外提取为 `--additive-field`。复用不再平滑或边缘引导，也不默认重复传旧 `--mask`。

要精确复现，保留原来的 `mode`、使用 `--strength 1` 和 `--field-strength 1`，不重复覆盖；若原场允许减光，复用须加 `--mode signed`，否则默认 `add-only` 会截去负 EV。示例中的 `--field-strength 0.7` 同时调弱这份 NPZ 的漫反射新增贡献与加性光；若 NPZ 来自双灯合成，它只能整体调节。分别调主灯与轮廓灯仍需各源控制图或各源 NPZ。新 NPZ 记录原件哈希，合成器会拒绝复用到其他原件；兼容读取的旧 NPZ 没有哈希时会记录未验证，不能自行声称属于当前照片。

| 选项 | 含义 |
| --- | --- |
| `--field gain.png` | 可重复传入增益场；也可读取同时含 EV 与加性光的原尺寸 NPZ |
| `--field-strength 1` | 每份 field 的线性新增贡献权重，范围 `0..4`；读取 NPZ 时也作用于其加性光。提供时与所有 field 一一对应，默认各为 `1` |
| `--additive-field highlights.png` | 可重复传入黑底加性 RGB 图；不接受 NPZ，复用 NPZ 用 `--field` |
| `--additive-strength 1` | 每份加性图的权重，范围 `0..4`；零为关闭。提供时与所有 additive-field 一一对应，默认各为 `1` |
| `--additive-scale 0.2` | 代码 `255` 代表的绝对线性 sRGB 新增值，范围 `0..4` |
| `--additive-smooth 0.8` | 加性图独立平滑尺度，单位为控制图像素，范围 `0..64` |
| `--additive-texture 0` | `0..1` 的原片局部亮度对比调制，默认关闭；只用于新加性图，NPZ 已含保存结果 |
| `--strength 1` | 范围 `0..2`；对增益场是 EV 倍率，对加性光是线性倍率；零关闭两类贡献，用于恒等核查 |
| `--ev-range 2` | 光照场编码的正端范围；必须和生成简报一致 |
| `--smooth 1.5` | 光照场平滑尺度，单位是控制图像素 |
| `--edge-radius 8` | 原图边缘引导的邻域尺度，按控制图像素计算 |
| `--epsilon 0.01` | 引导滤波的正则项；应结合边缘审查调整 |
| `--mask coverage.png` | 黑色保护、白色允许加光的覆盖蒙版 |
| `--mode add-only` | 默认只加光；`signed` 仅在允许减光时选用 |
| `--save-arrays` | 额外保存浮点 EV 与线性颜色增量 |

控制图或覆盖蒙版与原图显示画幅的宽高比相差超过 1% 时脚本拒绝合成。小于容差仍应检查几何对应，不能把容差称为精确配准；应优先生成相同比例，而非依靠容差拉伸。原图只提供像素与边缘引导，不被缩放、裁切或形变。

输出包括：

- `final.png`：原始显示尺寸的最终 sRGB 照片。
- `effect_overlay.png`：针对当前底图求出的 8 位 RGBA 正常混合效果层。
- `field_preview.png`、`coverage_preview.png`：处理后的控制场与覆盖缩览。
- `additive_preview.png`：存在加性光时输出的缩览，以 `0.2` 线性新增值归一化显示；更高值会显示白色，不能代替浮点数据。
- `report.json`：输入哈希、存储/显示尺寸、方向、色彩处理、参数与限制。`fields[]` 和 `additive_fields[]` 记录各输入与权重；`fields[]` 还记录 NPZ 是否含保存的加性光。仅一份增益输入时兼容保留 `field` 对象，其余为 `null`。
- `relight_arrays.npz`：使用 `--save-arrays` 时保存。`ev` 是原尺寸 float32 漫反射总 log2 增益；可选 `additive_linear` 是绝对正线性 RGB 新增光，两者均已含本次覆盖、强度与权重，并位于高光压缩前。`delta_linear` 是高光压缩后、8 位量化前相对原片的线性 RGB 差值。`source_sha256` 绑定原件。调强度或叠灯时通过 `--field` 复用 NPZ 的两类贡献，不把已压缩的 `delta_linear` 再当未处理灯光相加。

保持原始文件不动，工作文件存入独立运行目录。每个方案至少保留：原件标识、实际输入参考图、已审辅助图、生成光照场及其提示词、可选覆盖蒙版、最终原尺寸 PNG、透明效果层、合成参数与审图记录。需要再次调色或精确重算时，保留浮点增量数据；它和便于普通软件叠加的透明 PNG 用途不同。

透明效果层按正常模式叠在同一 sRGB 底图上，用于复现本次结果；它不是可随意换底的通用灯光素材。不同软件的混合色彩空间、透明度处理可能不同，导入后应核对。再次调强度优先重跑合成器，而非把已合成照片交给模型重绘。

按 [审图与参数对照](review-and-tests.md) 分别检查光效与保真。无法得到合格光照场时保留失败稿、说明局限；不自动切回 V1 整图重绘。用户选择 V1 后才使用旧流程，并报告分辨率及细节漂移。
