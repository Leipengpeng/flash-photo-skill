# 直接生成与原尺寸融合

两种交付共用原来的深度—语义材质—逐灯规划—完整Image编辑阶段。直接生成交付完整Image图；原尺寸融合继续从这张参考提取照明，并在原片显示网格上合成。已有同源参考可复用，换融合算法或强度不新增生图。

## 使用

需要Python、NumPy、SciPy、Pillow和OpenCV。脚本不调用Image服务，不下载模型，不训练网络。

```bash
python -m pip install -r scripts/requirements-native.txt
python scripts/native_fusion.py run --source original.jpg --reference full-flash.png --out work/native
```

默认F06。用户可明确指定F04，或同一参考同时交付两版：

```bash
python scripts/native_fusion.py run --source original.jpg --reference full-flash.png --methods F06 F04 --out work/compare
```

F06与F04来自用户认可的并列候选，未声称唯一最佳。输出目录不可覆盖原件或已完成结果；重做使用新目录。

## 固定实现

原图先按EXIF转正并经ICC转换为8-bit sRGB。在线性sRGB中，以精确像素足迹面积积分建立与完整参考同画幅的工作观测，不裁原图、不做RGB形变或普通插值放大。参考大于原片时也只缩小观测，最终底图仍是原片显示尺寸。宽高比允许最多1%的显式全画幅归一化网格容差；超过容差拒绝。尺寸接近不证明生成内容已精确配准。

共享R08观测：锚点步长2，以位置/原色找25个观测，3轮稳健拟合RGB照明常量及共享二维结构误差项，结构幅度限3工作像素。该项不是实测光流，也不移动RGB。

- **F06**：分别统计原片与参考在4/8/16工作像素尺度的区域光量比，取中位数；以原片大结构约束照明锚点，再添加通过梯度结构置信门控、限幅±0.25 EV的R08细光残差。
- **F04**：结构对应可信处多用R08照明；低置信处回退到sigma=8工作像素的区域光量估计，在EV层连续混合。

结构置信由原片/参考亮度梯度的归一化相关和能量计算；NCC在0.25–0.65间映射，平坦区域降权。边缘约束为原片log亮度sigma=3引导、锚点空间sigma=4、范围0.25 EV的三轮四方向递归。

原尺寸像素按自身颜色与位置寻找9个锚点并取中位数；F04的区域估计和置信度用空间近邻。位置尺度4工作像素、编码RGB尺度0.075、稳定项0.001、总EV限±3。

`final_linear = original_native_linear × 2^EV`

最后裁切显示范围、编码sRGB PNG。原尺寸RGB底图不模糊、不锐化、不移动。深度与材质图只参与上游生成规划，本地F04/F06不读取其数值。

函数由固定快照导出，计算函数AST一致性记录在 [来源校验](native-fusion-provenance.json)。主要入口为 `scripts/native_fusion.py`，纯算法在 `scripts/_native_fusion_core.py`；无开发机器绝对路径依赖。

## 文件与Delta

一次run保存：

- `original-sRGB.png`：转正、统一色管后的原尺寸底图。
- `pair.npz`：工作尺度`source_low_linear`、`reference_linear`和`raw_signed_delta_linear = reference - source_low`，含源文件哈希。
- `raw-delta-preview.png`：`round(128+127×raw_delta)`，128中性，仅作变化示意，不能当透明蒙版直接加回。
- 每个方法的`final.png`、`model.npz`、`report.json`、`delta_linear.npz`、`delta.json`和`delta-preview.png`。

原尺寸浮点Delta是`decode(delivered_final)-decode(original-sRGB)`，在原尺寸线性sRGB中相加可精确还原交付PNG；它与未处理的工作尺度差值分开。PNG无损压缩不代表光照修改没有信息损失，也不是RAW或广色域工作流。

用户要求一图一文件夹时，以原文件名（不含扩展名）命名。把原件、深度图、语义分割图、Image完整直闪、原始Delta示意、原尺寸结果放在同组；浮点Delta和模型可放数据子目录，保留路径索引。普通融合只交付选定的一版，不强制出算法实验全集。

## 调强度不重生图

```bash
python scripts/native_fusion.py apply --source original.jpg --model work/native/F06/model.npz --strength 0.8 --out work/weaker
```

模型保存未乘强度的基础场。精确重放采用首次run相同的`--strength`，默认1；0为原片恒等。强度范围0–3，是EV场缩放，不是已标定闪光功率。缓存绑定同一原文件字节哈希，不能静默换另一张或重编码副本。

## 验收

检查原件哈希、方向和显示尺寸、参考来源和工作尺度、raw Delta定义、输出是否有限、显示裁切和原尺寸Delta回加。脚本和数值夹具可用 `python -m unittest discover -s scripts -p test_native_fusion.py` 验证。

这不是自动审美评分。生成参考可能有重绘/错位，局部新高光或阴影也可能被融合抑制。灯光外观、边缘及细节由用户检查；助手不把程序复现通过写成效果最好。私人测试图不进入GitHub或安装包。
