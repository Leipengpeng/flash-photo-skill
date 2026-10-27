# 辅助图与最终提示词

## 通用约定

归一化区域框用 `[left, top, right, bottom]`，范围0–1，仅用于定位；复杂轮廓用文字和可见边界，不伪称精确掩膜。记录辅助图是 `generative-relative-estimate`，米制标定为 `none`。图中文字、图例和箭头放在独立简报中，默认不占用照片画幅。

每张辅助图只从原图生成。核对宽高比、对象轮廓、窗框和接触线，尺寸不匹配时不要直接声称逐像素对齐。必要时另存确定性重采样版本，核对边界并记录处理；重采样不等于无损对齐，无法可靠对应就重做或退回文字约束。全流程地图不是可作为数值深度/法线图读取的传感器数据。

同一原文件、方向、画幅/FOV与场景边界未变时，已审通过的 `depth.png` 和 `materials.png` 可缓存供多方案复用，记录原文件标识、地图路径/尺寸和审查结论。换了原图、裁切或发现误分就更新；未审图不能当作通过。缓存只覆盖几何/材质事实，每个光源的覆盖、遮挡、投影和反射路径仍重新规划，旧灯位路径图不通用。

高分辨率手机原图超出工具输出能力时，保留原始JPEG及EXIF不动；辅助图允许较低分辨率但须同宽高比、完整FOV及可对应边界，记录实际尺寸，只按归一化区域作定性参考，不能宣称原像素配准。最终照片若降分辨率也必须明说，不能把尺寸变小称为无损保留原图信息。不要为匹配模型尺寸先裁切用户原图。

## 深度图模板

```text
Use case: precise-object-edit; asset type: diagnostic relative-depth guide.
Input image 1 is the original photograph, the only geometry authority.
Generate a grayscale relative camera-depth estimate of exactly the same scene.
Keep the exact canvas, camera field of view, framing, perspective, silhouettes,
object locations and occlusion boundaries. White means near; black means far.
Use smooth depth changes on continuous surfaces, not shading or texture as depth.
Do not recolor according to the original light. No new objects, text or legend.
Observed near/mid/far ordering: {scene-specific observations}.
Transparent surfaces and transmitted scenes: {separate layer interpretation}.
For mirrors, separate the physical mirror plane from the reflected virtual scene.
Do not use virtual-image depth as a real source-to-subject distance.
This is a single-view inference guide, not metric depth or a measured 3D reconstruction.
```

深度表示按相机距离理解；离机闪光到物体的距离不一定等于相机深度。灯位偏离机轴时另做方向与遮挡推理，不直接把深度灰度套入平方反比。

镜面区补写 `surface_layer`（物理镜面位置、朝向证据与未知）和 `virtual_scene_layer`（镜中人物、房间、物件的表观层次）。看见镜中脸不表示该像素的物理第一表面是皮肤；照亮镜中人物对应“灯→真实主体→镜面→相机”，灯在镜中出现亮斑则对应“灯→镜面→相机”。两条路径分开，不把虚像当作现场接影面。

## 材质/反射可能性图模板

固定色板只是类别编码，不是颜色替换建议：

| 类别 | 颜色 |
|---|---|
| 不透明漫反射：墙、木、布、叶、普通石材 | 中灰 `#808080` |
| 皮肤 | 桃色 `#F2AA85` |
| 透明玻璃/清水可透射区域 | 青色 `#00C8DC` |
| 实际镜面 | 洋红 `#D040C8` |
| 抛光金属 | 黄色 `#F2CF30` |
| 湿润粗糙地面/水膜 | 蓝色 `#315EEA` |
| 可见发光体 | 橙色 `#F58B20` |
| 天空、远景或材质未知 | 深灰 `#303030` |

```text
Input image 1 is the original photograph. Generate a flat semantic material map
registered to its exact canvas, FOV, silhouettes, perspective and object positions.
Use ONLY these class colors: {palette above}. No text, no borders, no labels.
Scene-specific material regions: {observed regions}.
For a window, label the visible glass surface cyan; this does not relocate the
transmitted far scene to the glass plane. Keep solid mullions separate from glass.
For glasses and cups, preserve the exact real outlines; no invented mirrors.
Mark actual mirror surfaces as mirror, and describe reflected people/objects in a
separate virtual-scene layer rather than replacing the mirror class with skin/cloth.
Do not use source brightness as a material classifier. This is a material inference
guide, not actual ray tracing, a physical reflectance measurement or an albedo map.
```

同一像素出现多层透明内容时，类别图只显示指定表面层；简报补充透射层和未知。玻璃/金属不代表一定有可见闪光高光。

原图表面色不能直接映射成类别色。例如原图青色涂装的地板/墙/天花仍是不透明漫反射面，不能因为色板用青色编码玻璃就把它们标成玻璃；不透明纸杯与透明塑料杯也应依据边缘/透射证据分别判断。分类图修一次仍误导时，退回文字材质表，不把错误图当作强制条件。

## 分析简报字段

```text
source: 原图绝对路径与尺寸
locks: 人物/物件/文字、画幅/FOV、机位、透视锚点
depth_regions: 区域、相对远近、遮挡、物理表面/透射层/虚像层、把握与未知
materials: 区域、材质、粗糙/光滑推断、透射与反射条件
map_cache: 原文件与地图标识、尺寸/画幅、审查状态；本次生成或复用
light_sources[]: 每源的shape、effective_size、direction、elevation、aim、emission、modifier、strength、color_intent
response_by_source: 各源对前景/主体/近背景/远背景分别新增什么光
shadows_by_source: 有效发光面 -> 遮挡物 -> 接收面 -> 方向/半影/可见性
reflections_by_source: 有效发光面 -> 表面 -> 相机 -> 成立条件与高光形状
mirror_layers: 物理镜面与虚像分层；主体受光和光源镜中反射分别解释
ambient_policy: 环境曝光、白平衡与原有灯光保留策略
unknowns: 未标定距离/表面法线/曝光等
```

## 最终编辑模板

```text
Use case: lighting-weather. Edit the ORIGINAL photograph; do not recreate it.
Image 1: sole edit target and sole authority for identity, geometry and content.
Image 2: inferred relative-depth guide only, never a replacement scene.
Image 3: material-class guide only, its false colors must NOT appear in output.

Change only the added light contribution: {light_sources[] and each source's role}.
Independently specify each source's emitting shape, apparent size, position,
elevation, aim, flash/continuous emission, modifier, strength and color intent.
Preserve the clearly lit subject target from the approved strong-flash example
if one was supplied; it is a lighting reference only, never the edit base.
Preserve exactly the source framing, aspect ratio, field of view, camera position,
perspective anchors {list}, people, facial structure, pose, clothing, object count,
locations, original reflected/transmitted scene, surface detail and existing text.
Keep original ambient exposure and white balance; no global HDR or beauty retouch.
Preserve original blur, sensor noise and fine-detail uncertainty; no deblurring,
invented sharp fur/foliage/gravel, replacement labels or reconstructed tiny parts.

Depth-aware added light for EACH source: {foreground / subject / near / far background}.
Existing ambient shadows may be filled; occlusion is evaluated for each added source.
Source-specific shadow chains: {actual receivers, direction and source-dependent penumbra}.
Specular/transparent response: {conditional paths and projected emitter shape}.
For mirror scenes: {physical mirror plane / virtual scene / real-subject lighting paths}.
Face shadows if present: {hardness and transitions appropriate to the chosen source}.
Keep {far scene} at its original ambient light; no flash on distant skyline/sky.

No crop, outpainting, changed lens, moved furniture, changed face, added camera,
unrequested visible lighting hardware, blanket whitening, uniform glass dots,
fake flare, starbursts, new text, map colors, labels or invented shadow receiver.
Allow a reflected emitting shape only when the planned source-surface-camera path
supports it; do not invent stands, cameras or people in mirrors.
No invented wall or ceiling for bounce. Do not force hard shadows for a large source,
or visible motion/strobe differences merely to distinguish flash from continuous.
Uncalibrated distances, sizes and power are creative settings, not measured relighting.
Return one photograph with source canvas dimensions if supported.
```

删掉不存在的内容，如无人物就不写鼻影；把每个花括号替换成真实看图结论。若图像输入数量不足，删去未实际输入的图编号，并把图的已审结论改成文字。默认不添加第三张反射路径图；用户要可视化路径时另存示意图，并记录它对应的具体灯位。
