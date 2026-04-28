# FakeShield++ 示意图设计指南

> 本文档描述论文所需的所有示意图和系统结构图的设计规范，可直接用于指导 AI 画图工具（如 DALL-E、Midjourney、Stable Diffusion 等）生成插图。

---

## 📋 图表清单

| 编号 | 图表名称 | 类型 | 用途 | 目标位置 |
|------|---------|------|------|---------|
| Figure 1 | FakeShield++ 系统架构总览 | 系统结构图 | 展示整体框架和4个改进方向 | Introduction |
| Figure 2 | 自动路径映射工作流程 | 流程图 | 说明方向A的路径解析机制 | Design - Improvement 1 |
| Figure 3 | INT8量化原理示意 | 技术示意图 | 解释量化如何降低VRAM | Design - Improvement 2 |
| Figure 4 | 扩展DTG分类头结构 | 网络结构图 | 展示3类→6类的扩展方式 | Design - Improvement 3 |
| Figure 5 | CLIP特征引导掩码优化流程 | 流程图 | 说明异常图计算和融合过程 | Design - Improvement 4 |
| Figure 6 | VRAM对比柱状图 | 数据图表 | FP16 vs INT8 VRAM使用对比 | Evaluation |
| Figure 7 | 延迟对比柱状图 | 数据图表 | FP16 vs INT8推理延迟对比 | Evaluation |
| Figure 8 | IoU结果可视化 | 数据图表 | 每张图片的IoU分数 | Evaluation |
| Figure 9 | MFLM改进对比 | 数据图表 | Original vs CLIP-Refined IoU | Evaluation |

---

## 🎨 全局设计风格规范

### 配色方案
- **主色调**：深蓝 (#1a365d) + 科技蓝 (#3182ce) + 亮青 (#38b2ac)
- **辅助色**：橙色 (#ed8936) 用于高亮/对比，绿色 (#48bb78) 表示正向结果，红色 (#f56565) 表示问题/警告
- **背景**：白色或浅灰 (#f7fafc)
- **文字**：深灰 (#2d3748)

### 字体规范
- **英文**：Inter / Helvetica Neue / Arial
- **中文**：思源黑体 / 苹方
- **标题**：加粗，16-18pt
- **正文**：常规，12-14pt
- **代码/标签**：等宽字体 (JetBrains Mono / Fira Code)，10-12pt

### 图标风格
- 扁平化设计（Flat Design）
- 圆角矩形容器（border-radius: 8px）
- 细线边框（2px stroke）
- 统一使用 Material Design Icons 或 Lucide Icons 风格

---

## 🏗️ Figure 1: FakeShield++ 系统架构总览

### 画面布局
```
┌─────────────────────────────────────────────────────────────────┐
│                    FakeShield++ Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Input Image │───▶│   DTE-FDM    │───▶│    MFLM      │      │
│  │              │    │  (Detection) │    │ (Localization)│      │
│  └──────────────┘    └──────┬───────┘    └──────┬───────┘      │
│                             │                   │               │
│              ┌──────────────┴──────────────┐    │               │
│              │     Improvement Modules     │    │               │
│              ├─────────────────────────────┤    │               │
│              │  A: Auto Path Mapping       │────┘               │
│              │  B: INT8 Quantization       │                    │
│              │  C: Extended DTG (6-class)  │                    │
│              │  D: CLIP-Guided Refinement  │───────────────────▶│
│              └─────────────────────────────┘                    │
│                                                                 │
│  Output: FAKE/REAL + Explanation + Localization Mask            │
└─────────────────────────────────────────────────────────────────┘
```

### AI 画图提示词（Prompt）
```
A professional system architecture diagram for an AI-based image forgery detection system called "FakeShield++". 

The diagram should show:
1. Left side: An input image icon (a photo with a magnifying glass)
2. Center: Two main processing blocks - "DTE-FDM" (detection module with LLaVA icon) and "MFLM" (localization module with segmentation mask icon)
3. Bottom: Four improvement modules shown as colored cards:
   - "A: Auto Path Mapping" (green, with file/folder icon)
   - "B: INT8 Quantization" (blue, with chip/memory icon)
   - "C: Extended DTG" (orange, with classification/tag icon)
   - "D: CLIP-Guided Refinement" (purple, with eye/vision icon)
4. Right side: Output showing three elements - FAKE/REAL label, text explanation bubble, and a segmentation mask overlay

Style: Clean, modern, flat design with blue and white color scheme. Use arrows to show data flow. Include subtle gradient backgrounds for each module. Professional technical diagram suitable for an academic paper.

Aspect ratio: 16:9
```

### 关键元素说明
- **DTE-FDM 模块**：标注 "LLaVA-v1.5-13B + DTG"
- **MFLM 模块**：标注 "GLaMM + SAM"
- **改进模块 A**：用文件夹+搜索图标表示路径自动解析
- **改进模块 B**：用芯片/内存条图标表示量化
- **改进模块 C**：用标签/分类图标表示扩展分类
- **改进模块 D**：用眼睛/视觉图标表示 CLIP 引导

---

## 🔀 Figure 2: 自动路径映射工作流程

### 画面布局
```
┌─────────────────────────────────────────────────────────────┐
│              Automatic Path Mapping Pipeline                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  DTE-FDM Output          fix_image_path.py         MFLM     │
│  ┌─────────────┐        ┌─────────────────┐       ┌──────┐ │
│  │ {           │        │ 1. Extract      │       │      │ │
│  │  "path":    │───────▶│    filename     │──────▶│ Mask │ │
│  │  "/home/... │        │ 2. Search local │       │      │ │
│  │  /0281.jpg" │        │    directory    │       └──────┘ │
│  │ }           │        │ 3. Replace path │                │
│  └─────────────┘        └─────────────────┘                │
│                                                             │
│  Before: /home/user/FakeShield/dataset/casia1+/0281.jpg     │
│  After:  ./dataset/casia1_plus/Tp/0281.jpg  ✅              │
└─────────────────────────────────────────────────────────────┘
```

### AI 画图提示词
```
A clean flowchart diagram showing an automatic file path mapping process for a machine learning pipeline.

The diagram should have three main stages connected by arrows:
1. Left: A JSON file icon showing a path "/home/user/dataset/image.jpg" with a red X mark indicating the path is broken
2. Center: A processing box labeled "fix_image_path.py" with three steps inside:
   - Step 1: Extract filename (show a magnifying glass over "image.jpg")
   - Step 2: Search local directory (show a folder tree with search icon)
   - Step 3: Replace path (show text replacement animation)
3. Right: A segmentation mask output icon with a green checkmark

Below the flowchart, show a before/after comparison:
- Before: "/home/user/FakeShield/dataset/casia1+/0281.jpg" (red, crossed out)
- After: "./dataset/casia1_plus/Tp/0281.jpg" (green, with checkmark)

Style: Minimalist flat design, blue and white color scheme, clear typography, suitable for technical documentation.

Aspect ratio: 16:9
```

---

## 💾 Figure 3: INT8 量化原理示意

### 画面布局
```
┌─────────────────────────────────────────────────────────────┐
│              INT8 Quantization for DTE-FDM                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  FP16 Weights                    INT8 Weights               │
│  ┌─────────────────┐            ┌─────────────────┐        │
│  │ 16-bit floating │            │  8-bit integer  │        │
│  │    point        │──Quantize─▶│  + scale factor │        │
│  │                 │            │                 │        │
│  │ 2 bytes/param   │            │ 1 byte/param    │        │
│  │ 6.94 GB VRAM    │            │ 3.98 GB VRAM    │        │
│  └─────────────────┘            └─────────────────┘        │
│                                                             │
│  Trade-off:                                                 │
│  ✅ 43% VRAM reduction                                      │
│  ⚠️ 48% latency increase (dequantization overhead)          │
│  ✅ Accuracy preserved (identical detection results)         │
└─────────────────────────────────────────────────────────────┘
```

### AI 画图提示词
```
A technical illustration showing INT8 quantization for a large language model.

The diagram should show:
1. Left side: A large matrix/grid labeled "FP16 Weights" with 16-bit floating point numbers (e.g., "0.12345", "-0.67890"), colored in deep blue. Show "2 bytes per parameter" and "6.94 GB VRAM" labels.
2. Center: A quantization arrow labeled "bitsandbytes INT8" showing the conversion process. Include a small formula: "W_int8 = round(W_fp16 * 127 / max_abs)"
3. Right side: A smaller matrix/grid labeled "INT8 Weights" with 8-bit integers (e.g., "15", "-87", "42"), colored in lighter blue. Show "1 byte per parameter" and "3.98 GB VRAM" labels.
4. Bottom: A trade-off summary box showing:
   - Green checkmark: "43% VRAM reduction"
   - Orange warning: "48% latency increase"
   - Green checkmark: "Accuracy preserved"

Style: Clean technical diagram with a gradient blue color scheme. Use grid patterns to represent weight matrices. Include subtle circuit/chip design elements in the background.

Aspect ratio: 16:9
```

---

## 🏷️ Figure 4: 扩展 DTG 分类头结构

### 画面布局
```
┌─────────────────────────────────────────────────────────────┐
│         Extended Domain Tag Generator (3 → 6 classes)        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ResNet-50 Backbone          Classification Head            │
│  ┌─────────────────┐        ┌──────────────────────────┐   │
│  │                 │        │  Original (3-class)      │   │
│  │  Conv Layers    │        │  ├─ AIGC                 │   │
│  │  (2048-dim)     │───────▶│  ├─ DeepFake             │   │
│  │                 │        │  └─ Photoshop            │   │
│  └─────────────────┘        ├──────────────────────────┤   │
│                             │  Extended (6-class)      │   │
│                             │  ├─ AIGC                 │   │
│                             │  ├─ DeepFake             │   │
│                             │  ├─ Photoshop            │   │
│                             │  ├─ SD_inpaint    [NEW]  │   │
│                             │  ├─ ControlNet    [NEW]  │   │
│                             │  └─ Midjourney    [NEW]  │   │
│                             └──────────────────────────┘   │
│                                                             │
│  Weight Initialization:                                     │
│  W_new[3:6] = W_orig[0] + N(0, 0.01)  ← copy from AIGC    │
└─────────────────────────────────────────────────────────────┘
```

### AI 画图提示词
```
A neural network architecture diagram showing the extension of a classification head from 3 classes to 6 classes.

The diagram should show:
1. Left: A ResNet-50 backbone represented as a stack of convolutional layers (colored blocks), outputting a 2048-dimensional feature vector
2. Center: An arrow labeled "Global Average Pooling" connecting the backbone to the classification head
3. Right: Two classification heads shown side by side or stacked:
   - Original head (top): 3 output neurons labeled "AIGC", "DeepFake", "Photoshop" in blue
   - Extended head (bottom): 6 output neurons, with the original 3 in blue and 3 new ones ("SD_inpaint", "ControlNet", "Midjourney") in orange with "NEW" badges
4. Bottom: A weight initialization formula showing: "W_new = W_AIGC + noise" with a small illustration of copying weights from the AIGC class to new classes

Style: Clean neural network diagram with rounded rectangles for layers. Use blue for original components and orange for new/extended components. Include subtle gradient fills and shadow effects.

Aspect ratio: 16:9
```

---

## 👁️ Figure 5: CLIP 特征引导掩码优化流程

### 画面布局
```
┌─────────────────────────────────────────────────────────────┐
│          CLIP Feature-Guided Mask Refinement                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input Image     CLIP Features     Anomaly Map    Refined   │
│  ┌─────────┐    ┌─────────────┐   ┌───────────┐   Mask     │
│  │         │    │ f_1 f_2 ... │   │           │  ┌───────┐ │
│  │  Image  │───▶│ f_n (patch) │──▶│ Heatmap   │─▶│ Mask  │ │
│  │         │    │             │   │           │  └───────┘ │
│  └─────────┘    └─────────────┘   └───────────┘           │
│                                                             │
│  Key Steps:                                                 │
│  1. Extract CLIP features from layer -2 (spatial)           │
│  2. Compute cosine similarity to global average             │
│  3. Anomaly map = 1 - similarity                            │
│  4. Fuse: M_refined = (1-α)·M_orig + α·A                    │
└─────────────────────────────────────────────────────────────┘
```

### AI 画图提示词
```
A detailed flowchart showing a CLIP feature-guided mask refinement process for image forgery localization.

The diagram should have four stages connected by arrows:
1. **Input Image** (left): Show an image with a tampered region (e.g., a spliced object). The image should have a subtle red outline around the tampered area.
2. **CLIP Feature Extraction** (center-left): Show the CLIP vision tower extracting patch-level features. Represent this as a grid of feature vectors (small colored squares), with each patch having a different color based on its feature similarity.
3. **Anomaly Map Generation** (center-right): Show a heatmap where the tampered region is highlighted in red/orange (high anomaly score) and the rest of the image is blue/green (low anomaly). Include a small color bar legend.
4. **Refined Mask** (right): Show the final binary segmentation mask (white tampered region on black background), with a green checkmark.

Below the main flowchart, show the fusion formula:
"M_refined = (1-α) × M_original + α × Anomaly_Map"
with a visual representation of the blending process.

Style: Professional technical diagram with a clean layout. Use a warm-to-cool color gradient for the anomaly map. Include mathematical notation in a clean sans-serif font.

Aspect ratio: 16:9
```

---

## 📊 Figure 6-9: 数据图表（已生成）

这些图表已经通过 matplotlib 生成，位于 `mypaper/figures/` 目录下：

| 文件名 | 描述 |
|--------|------|
| `vram_comparison.png` | FP16 (6.94GB) vs INT8 (3.98GB) VRAM 对比柱状图 |
| `latency_comparison.png` | FP16 (35.18s) vs INT8 (52.26s) 延迟对比柱状图 |
| `iou_results.png` | 4张测试图片的 IoU 分数柱状图 |
| `mflm_improvement.png` | Original vs CLIP-Refined IoU 对比分组柱状图 |

### 数据图表风格规范（如需重新生成）
- **配色**：FP16=蓝色 (#3182ce)，INT8=橙色 (#ed8936)，Original=灰色 (#a0aec0)，CLIP-Refined=绿色 (#48bb78)
- **字体**：Arial/Helvetica，标题 14pt，标签 12pt
- **网格**：浅灰色虚线网格，y 轴从 0 开始
- **误差线**：如有多次实验，添加标准差误差线
- **图例**：右上角或底部居中
- **分辨率**：300 DPI，保存为 PNG

---

## 🎯 画图 AI 使用建议

### DALL-E 3
- 使用上述完整的英文 prompt
- 指定 "technical diagram" 或 "infographic" 风格
- 如生成结果不理想，可以分多次生成各个模块，然后用 Figma/Canva 拼接

### Midjourney
- 添加 `--v 6 --ar 16:9 --style raw` 参数
- 使用 `--no text, watermark, signature` 避免不必要的文字
- Midjourney 对技术图表的支持较弱，建议用于生成概念性插图

### Stable Diffusion
- 使用 ControlNet 的 Canny 或 Depth 模型控制布局
- 先用 draw.io 或 Excalidraw 绘制草图，然后用 img2img 生成
- 推荐模型：`dreamshaper` 或 `realisticVision`

### 手动编辑工具
- **draw.io** / **Excalidraw**：快速绘制流程图和架构图
- **Figma**：精细化调整和统一风格
- **Inkscape**：矢量图编辑，适合学术论文
- **Canva**：模板化设计，快速出图

---

## 📐 尺寸和分辨率规范

| 用途 | 宽度 | 分辨率 | 格式 |
|------|------|--------|------|
| 论文插图（单栏） | 3.5 inches (89mm) | 300 DPI | PNG/PDF |
| 论文插图（双栏） | 7 inches (178mm) | 300 DPI | PNG/PDF |
| 演讲 PPT | 1920px | 72-150 DPI | PNG |
| GitHub README | 800px | 72 DPI | PNG |

---

_文档版本：v1.0 | 最后更新：2025-04-28_
