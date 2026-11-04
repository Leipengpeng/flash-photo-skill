# 原尺寸融合

准备原始照片和同一画幅的完整打光图。每种布光只需一张模型成图，可复用已有参考。

```bash
python -m pip install -r scripts/requirements-native.txt
python scripts/native_fusion.py run --source original.jpg --reference relit.jpg --out work/native
```

脚本最后一行 JSON 的 `final` 字段指向原尺寸 PNG。运行目录内同时保存规范化原图、相减预览、融合后 Delta 和可重放的照明数据。原尺寸以照片按 EXIF 转正后的显示宽高为准。

## 照明怎么回到原片

1. 将原片与模型成图转为线性 RGB，按像素覆盖面积建立同画幅的工作观测，记录有正负的相减结果。
2. 从多个尺度估计区域光量，用原片边缘约束光照场，再加入与原片结构吻合的局部光照变化。
3. 按原片像素的颜色和位置采样照明增益，应用到原片的 RGB 上，保存原尺寸 PNG。

深度和材质图用于生成阶段规划布光。融合读取原片和完整打光图。更换灯光时从原片出发。只调整融合强度时，可用返回 JSON 的 `model` 路径本地重放。

```bash
python scripts/native_fusion.py apply --source original.jpg --model path/to/model.npz --strength 0.7 --out work/softer
```
