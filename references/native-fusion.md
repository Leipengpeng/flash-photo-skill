# 原尺寸融合

`--source` 使用完整原片文件，路径指向原始附件或用户指定的本地文件。`--reference` 使用同一画幅的完整打光图。可用缩小副本生成参考，也可复用已有参考。

```bash
python -m pip install -r scripts/requirements-native.txt
python scripts/native_fusion.py run --source original.jpg --reference relit.jpg --out work/native
```

## 文件交付

脚本成功后的最后一行 JSON 提供 `final`、`format`、`size`、`bit_depth` 和 `sha256`。确认 `final` 指向的文件存在且确实是 PNG，`size` 与原片按 EXIF 转正后的宽高一致，位深符合输入。8 位照片输出 8 位 RGB PNG。16 位不透明 RGB 或灰度 sRGB PNG 和单页 TIFF 输出 16 位 RGB PNG。两种输出均采用无损编码。

直接把 `final` 文件作为附件交付，或提供可打开的绝对路径。需要另存成品时按字节复制，可用 `sha256` 核验。平台只提供可能转码的图片卡片时，将成品 PNG 装进 ZIP 作为文件交付。缩览、模型直出和 Delta 预览分别标注用途。

原片不可访问、融合运行失败或格式不受支持时，说明未完成原尺寸融合及原因。部分 16 位 TIFF 压缩或嵌入的色彩配置需要额外解码或色彩转换组件，可按脚本提示安装。相机 RAW 或当前环境不支持的输入可预先导出为 sRGB PNG，并保留目标位深。运行目录还保存规范化原图、相减预览、融合后 Delta 和可重放的照明数据。

## 照明怎么回到原片

1. 将原片与模型成图转为线性 RGB，按像素覆盖面积建立同画幅的工作观测，记录有正负的相减结果。
2. 从多个尺度估计区域光量，用原片边缘约束光照场，再加入与原片结构吻合的局部光照变化。
3. 按原片像素的颜色和位置采样照明增益，应用到原片的 RGB 上，保存原尺寸 PNG。

深度和材质图用于生成阶段规划布光。融合读取原片和完整打光图。更换灯光时从原片出发。只调整融合强度时，可用返回 JSON 的 `model` 路径本地重放。

```bash
python scripts/native_fusion.py apply --source original.jpg --model path/to/model.npz --strength 0.7 --out work/softer
```
