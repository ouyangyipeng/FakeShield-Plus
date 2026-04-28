# FakeShield 改进实验计划

## 项目概述

基于 FakeShield (ICLR 2025) 论文复现工作，在实现过程中发现多个问题并提出改进方案。本计划包含 4 个改进方向，每个方向均有明确的问题定义、改进方法、实验设计和预期结果。

**论文信息**：
- 标题：FakeShield: Explainable Image Forgery Detection and Localization via Multi-modal Large Language Models
- 会议：ICLR 2025
- 作者：Zhipei Xu, Xuanyu Zhang, Runyi Li, Zecheng Tang, Qing Huang, Jian Zhang (Peking University)

**当前环境**：
- GPU: 4× NVIDIA A100 (80GB)
- PyTorch: 1.13.0+cu117
- transformers: 4.28.0
- mmcv-full: 1.4.7

---

## 改进方向 A：Pipeline 鲁棒性改进

### 问题定义

在复现过程中发现，DTE-FDM 输出的 JSONL 文件中包含的图片路径是训练数据的绝对路径（如 `/data03/xzp/dataset/ps/CASIA1+/image/...`），导致 MFLM 模块无法直接读取图片进行定位。这是一个严重的 pipeline 鲁棒性问题：

1. **路径依赖问题**：DTE-FDM 输出硬编码了训练环境的路径
2. **部署困难**：在不同环境中部署时需要手动修改路径
3. **端到端失败**：原始 `run_pipeline.sh` 无法直接运行

### 改进方法

#### A.1 路径自动映射模块

在 pipeline 中添加 `PathResolver` 模块，自动处理路径映射：

```python
# 伪代码
class PathResolver:
    def __init__(self, base_path_mapping):
        self.mapping = base_path_mapping  # {train_prefix: local_prefix}
    
    def resolve(self, dte_fdm_output):
        """自动解析 DTE-FDM 输出中的图片路径"""
        original_path = dte_fdm_output.get("image", "")
        for train_prefix, local_prefix in self.mapping.items():
            if original_path.startswith(train_prefix):
                return original_path.replace(train_prefix, local_prefix)
        return original_path
```

#### A.2 改进的流水线脚本

修改 `run_pipeline.sh`，添加自动路径解析：

```bash
# 改进后的流水线
python -m llava.serve.cli ... --output-path ${DTE_FDM_OUTPUT}

# 新增：自动路径修复
python scripts/fix_image_path.py \
    --input ${DTE_FDM_OUTPUT} \
    --output ${DTE_FDM_OUTPUT_FIXED} \
    --local-image-dir ./playground/image

python ./MFLM/cli_demo.py \
    --DTE-FDM-output ${DTE_FDM_OUTPUT_FIXED} \
    ...
```

### 实验设计

| 实验组 | 描述 | 指标 |
|--------|------|------|
| Baseline | 原始 pipeline（手动指定 --image-path） | 端到端成功率 |
| Ours-A | 改进后的 pipeline（自动路径映射） | 端到端成功率 |

**测试数据集**：使用 playground 中的 4 张测试图片

**评估指标**：
- 端到端成功率（成功生成 mask 的图片数 / 总图片数）
- 用户干预次数

### 预期结果

改进后的 pipeline 应实现 100% 端到端成功率，无需用户手动指定图片路径。

---

## 改进方向 B：推理效率优化

### 问题定义

DTE-FDM 基于 LLaVA-v1.5-13B（130 亿参数），推理速度慢、显存占用高：

1. **推理延迟**：单张图片推理约需 10-30 秒
2. **显存占用**：FP16 推理需要约 26GB 显存
3. **部署限制**：无法在消费级 GPU 上运行

### 改进方法

#### B.1 INT8 量化

使用 `bitsandbytes` 库进行 INT8 量化：

```python
from transformers import BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
)

model = LlavaLlamaForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    device_map="auto",
)
```

#### B.2 INT4 量化（可选）

进一步压缩到 INT4：

```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
```

### 实验设计

| 实验组 | 量化方式 | 显存占用 | 推理速度 | 检测准确率 |
|--------|----------|----------|----------|------------|
| Baseline | FP16 | ~26GB | 基准 | 基准 |
| Ours-B-INT8 | INT8 | ~13GB | ? | ? |
| Ours-B-INT4 | INT4 | ~7GB | ? | ? |

**测试数据集**：CASIA1+ 子集（100 张图片）

**评估指标**：
- 显存占用（GB）
- 推理速度（秒/张）
- 检测准确率（ACC, F1）
- 解释质量（CSS）

### 预期结果

- INT8 量化：显存减少 ~50%，速度提升 ~30%，准确率下降 <2%
- INT4 量化：显存减少 ~70%，速度提升 ~50%，准确率下降 <5%

---

## 改进方向 C：跨域泛化能力改进

### 问题定义

DTG（Domain Tag Generator）仅支持 3 类域标签：
1. PhotoShop (PS)
2. DeepFake (DF)
3. AIGC-Editing (AIGC)

对于新型伪造方式（如 ControlNet Inpainting、SDXL Inpainting），DTG 无法正确分类，导致 DTE-FDM 缺乏正确的域引导。

### 改进方法

#### C.1 扩展 DTG 分类

将 DTG 从 3 类扩展到 6 类：

| 类别 | 原始 | 新增 |
|------|------|------|
| PS | ✅ | |
| DeepFake | ✅ | |
| AIGC-Editing | ✅ | |
| ControlNet-Inpaint | | ✅ |
| SDXL-Inpaint | | ✅ |
| GAN-based | | ✅ |

#### C.2 零样本域分类（备选方案）

使用 CLIP 进行零样本域分类，无需重新训练 DTG：

```python
from transformers import CLIPModel, CLIPProcessor

clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

domain_labels = [
    "Photoshop manipulated image",
    "DeepFake manipulated face",
    "AIGC edited image",
    "ControlNet inpainted image",
    "SDXL inpainted image",
    "GAN generated image",
]

def zero_shot_domain_classification(image):
    inputs = clip_processor(
        text=domain_labels,
        images=image,
        return_tensors="pt",
        padding=True,
    )
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    return domain_labels[probs.argmax()]
```

### 实验设计

| 实验组 | DTG 类型 | 测试数据集 | 指标 |
|--------|----------|------------|------|
| Baseline | 3 类 DTG | ControlNet/SDXL Inpaint | ACC, F1 |
| Ours-C-extended | 6 类 DTG | ControlNet/SDXL Inpaint | ACC, F1 |
| Ours-C-zeroshot | CLIP 零样本 | ControlNet/SDXL Inpaint | ACC, F1 |

**测试数据集**：
- ControlNet Inpainting: 500 张
- SDXL Inpainting: 500 张

**评估指标**：
- 域分类准确率
- 检测准确率（ACC, F1）
- 定位精度（IoU, F1）

### 预期结果

扩展 DTG 后，在新型伪造数据上的检测准确率应提升 10-20%。

---

## 改进方向 D：MFLM 定位精度改进

### 问题定义

MFLM 的定位严重依赖 DTE-FDM 的文本描述 $O_{det}$。当 DTE-FDM 输出的文本描述不准确时（如位置描述错误、篡改类型判断错误），MFLM 的定位精度会显著下降。

论文附录 A.3 中也提到了这个问题，但他们发现引入 error correction 机制并不能改善结果（Table 8）。

### 改进方法

#### D.1 图像特征直接引导

在 MFLM 中引入图像特征直接引导，减少对文本的依赖：

**原始 MFLM**：
```
输入: {O_det, T_img} → TCM → h_<SEG> → SAM → M_loc
```

**改进 MFLM**：
```
输入: {O_det, T_img, E_visual} → Enhanced TCM → h_<SEG> → SAM → M_loc
```

其中 $E_{visual}$ 是从 CLIP 图像编码器提取的视觉特征，直接与 SAM 的中间特征融合。

#### D.2 实现方案

```python
# 伪代码：改进的 TCM
class EnhancedTamperComprehensionModule(nn.Module):
    def __init__(self, original_tcm):
        super().__init__()
        self.tcm = original_tcm
        # 新增：视觉特征融合层
        self.visual_fusion = nn.Sequential(
            nn.Linear(1024 + 1024, 1024),  # 拼接文本+视觉特征
            nn.GELU(),
            nn.Linear(1024, 1024),
        )
    
    def forward(self, text_tokens, image_tokens, visual_features):
        # 原始 TCM 输出
        text_embedding = self.tcm(text_tokens, image_tokens)
        # 新增：视觉特征融合
        combined = torch.cat([text_embedding, visual_features], dim=-1)
        enhanced = self.visual_fusion(combined)
        return enhanced
```

### 实验设计

| 实验组 | MFLM 输入 | 测试数据集 | IoU | F1 |
|--------|-----------|------------|-----|-----|
| Baseline | {O_det, T_img} | CASIA1+ | 0.54 | 0.60 |
| Ours-D | {O_det, T_img, E_visual} | CASIA1+ | ? | ? |
| Baseline | {O_det, T_img} | IMD2020 | 0.50 | 0.57 |
| Ours-D | {O_det, T_img, E_visual} | IMD2020 | ? | ? |

**测试数据集**：
- CASIA1+: 920 张篡改图片
- IMD2020: 2010 张篡改图片

**评估指标**：
- IoU（Intersection over Union）
- Pixel-level F1

### 预期结果

引入图像特征直接引导后，IoU 应提升 3-8%，特别是在 DTE-FDM 文本描述不准确的情况下改善更明显。

---

## 实验时间表

| 阶段 | 内容 | 预计时间 |
|------|------|----------|
| 阶段 1 | 方向 A：实现路径映射模块 + 实验 | 1 天 |
| 阶段 2 | 方向 B：实现量化 + 实验 | 2 天 |
| 阶段 3 | 方向 C：扩展 DTG + 实验 | 2 天 |
| 阶段 4 | 方向 D：改进 MFLM + 实验 | 2 天 |
| 阶段 5 | 论文撰写 | 2 天 |
| 阶段 6 | 论文修改和完善 | 1 天 |

**总计**：约 10 天

---

## 论文结构（IEEE 会议格式）

```
标题：FakeShield++: Enhanced Explainable Image Forgery Detection and Localization
      with Robust Pipeline and Improved Generalization

摘要 (Abstract)
1. 引言 (Introduction)
   - FakeShield 的贡献与局限性
   - 本文的 4 个改进方向
2. 相关工作 (Related Work)
   - IFDL 方法
   - MLLM 在伪造检测中的应用
3. 方法 (Methodology)
   3.1 Pipeline 鲁棒性改进 (方向 A)
   3.2 推理效率优化 (方向 B)
   3.3 跨域泛化能力改进 (方向 C)
   3.4 MFLM 定位精度改进 (方向 D)
4. 实验 (Experiments)
   4.1 实验设置
   4.2 方向 A 实验结果
   4.3 方向 B 实验结果
   4.4 方向 C 实验结果
   4.5 方向 D 实验结果
   4.6 消融实验
5. 结论 (Conclusion)
参考文献 (References)
```

---

## 风险与注意事项

1. **数据集获取**：CASIA1+、IMD2020 等数据集需要从论文提供的链接下载
2. **训练时间**：方向 C 和 D 需要微调模型，可能需要较长时间
3. **显存限制**：方向 B 的量化实验需要确保 bitsandbytes 与 PyTorch 1.13 兼容
4. **国内网络**：所有下载需使用国内镜像源

---

## 快速启动命令

```bash
cd FakeShield
source venv/bin/activate

# 方向 A：Pipeline 鲁棒性
python scripts/fix_image_path.py --help

# 方向 B：量化推理
python scripts/quantized_inference.py --help

# 方向 C：扩展 DTG
python scripts/train_extended_dtg.py --help

# 方向 D：改进 MFLM
python scripts/train_enhanced_mflm.py --help
```
