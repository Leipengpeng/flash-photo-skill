# 辅助图与出图提示词

深度图和材质图分别从原图生成。保留全画幅、对象轮廓和遮挡关系。图例和文字放在分析记录里。

## 相对深度图

```text
Generate a grayscale relative-depth map of the supplied photograph.
Keep the source framing, camera perspective, silhouettes and occlusion boundaries.
White is near and black is far. Follow scene geometry rather than shading or texture.
Observed near, middle and far regions: {regions}.
Describe window glass and the scene beyond it as separate layers. Separate mirror
surfaces from reflected scenes. Return a clean map without labels.
```

核对前、中、后景和接触线。离机灯按灯的位置判断距离。玻璃位置与玻璃后景物分开。镜中反光分别考虑「灯 → 镜面 → 相机」和「灯 → 真实主体 → 镜面 → 相机」。

## 语义材质图

| 类别 | 颜色 |
| --- | --- |
| 墙、木、布、叶、普通石材 | 中灰 `#808080` |
| 皮肤 | 桃色 `#F2AA85` |
| 透明玻璃或水 | 青色 `#00C8DC` |
| 镜面 | 洋红 `#D040C8` |
| 抛光金属 | 黄色 `#F2CF30` |
| 湿地或水膜 | 蓝色 `#315EEA` |
| 发光体 | 橙色 `#F58B20` |
| 远景或未知材质 | 深灰 `#303030` |

```text
Generate a flat material-class map for the supplied photograph using {palette}.
Keep the framing, perspective, object positions and boundaries of the source.
Classify materials by surface evidence and use the observed regions: {regions}.
Mark the actual glass or mirror surface and describe its visible contents separately.
Return the map without labels or outlines.
```

## 整图编辑模板

把占位符替换为该场景的观察结果，填写实际传入的辅助图。

```text
Edit the original photograph to add {requested lighting}.
Use the relative-depth and material maps only to plan lighting.
Preserve the source canvas, camera perspective, identity, pose, objects, textures,
text, and visible reflections. Keep the ambient light in the far background.
For each light, follow its shape, apparent size, position, aim, coverage,
strength and color: {light plan}.
Light response by depth: {foreground, subject, near and far background}.
Place shadows on their actual receiving surfaces, with direction and softness
set by each source: {shadow plan}.
Apply material-specific reflections where the light-to-surface-to-camera path
allows them: {reflection plan}.
Return one relit photograph. Keep lighting equipment and guide colors out of frame.
```
