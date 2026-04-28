# FakeShield++: Enhanced Robustness and Efficiency for Multimodal Image Forgery Detection and Localization

> Course project for Multimedia Technology. Based on FakeShield (ICLR 2025), we systematically reproduce and improve the framework along 4 dimensions: pipeline robustness, inference efficiency, cross-domain generalization, and localization evaluation.

## 📋 Overview

This repository contains:
- **FakeShield/**: Original FakeShield codebase with our improvements
- **mypaper/**: IEEE conference paper (LaTeX source)
- **plans/**: Experiment planning documents

## 🛠️ Improvements

### Direction A: Pipeline Robustness
- **Problem**: DTE-FDM outputs hardcoded absolute paths, causing MFLM to fail in cross-environment deployment
- **Solution**: Automatic path mapping module (`scripts/fix_image_path.py`)
- **Result**: 100% end-to-end success rate without manual `--image-path` specification

### Direction B: Inference Efficiency
- **Problem**: 13B-parameter DTE-FDM requires 6.94 GB GPU VRAM (FP16)
- **Solution**: INT8 quantization via bitsandbytes
- **Result**: 43% VRAM reduction (3.98 GB per GPU) with identical detection accuracy

### Direction C: Cross-domain Generalization
- **Problem**: DTG only supports 3 forgery classes (AIGC/DeepFake/Photoshop)
- **Solution**: Extended DTG with 6 classes (+SD_inpaint, ControlNet, Midjourney)
- **Result**: Fine-tuned on SD_inpaint dataset, 75% accuracy on small test set

### Direction D: Localization Evaluation
- **Problem**: No systematic IoU evaluation framework for MFLM
- **Solution**: IoU evaluation framework + CLIP feature-guided mask refinement
- **Result**: Mean IoU 0.79 on synthetic evaluation

## 📁 Project Structure

```
├── FakeShield/              # Original FakeShield + improvements
│   ├── scripts/             # Our improvement scripts
│   │   ├── fix_image_path.py       # Direction A: path mapping
│   │   ├── run_pipeline_improved.sh # Direction A: improved pipeline
│   │   ├── quantized_inference.py  # Direction B: INT8/INT4 benchmark
│   │   ├── extend_dtg.py           # Direction C: extended DTG training
│   │   ├── cross_domain_benchmark.py # Direction C: cross-domain eval
│   │   ├── evaluate_iou.py         # Direction D: IoU evaluation
│   │   └── mflm_improved.py        # Direction D: CLIP-guided refinement
│   ├── DTE-FDM/             # DTE-FDM module (modified builder.py)
│   ├── MFLM/                # MFLM module
│   └── playground/          # Test results
├── mypaper/                 # IEEE conference paper
│   ├── main.tex             # Main document
│   ├── sections/            # Individual section files
│   │   ├── abstract.tex
│   │   ├── introduction.tex
│   │   ├── original_paper.tex
│   │   ├── design.tex
│   │   ├── evaluation.tex
│   │   └── discussion.tex
│   └── references.bib       # Bibliography
└── plans/                   # Experiment planning
    ├── fakeshield-setup-plan.md
    └── fakeshield-improvement-plan.md
```

## 🚀 Quick Start

### Environment Setup
```bash
# See FakeShield/ROADMAP.md for detailed setup instructions
pip install -r FakeShield/requirements.txt
pip install -e FakeShield/DTE-FDM
```

### Run Improved Pipeline
```bash
cd FakeShield
bash scripts/run_pipeline_improved.sh
```

### Run Quantization Benchmark
```bash
cd FakeShield
python scripts/quantized_inference.py --quantization fp16
python scripts/quantized_inference.py --quantization int8
```

### Compile Paper
```bash
cd mypaper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## 📊 Key Results

| Metric | Original | Improved |
|--------|----------|----------|
| Pipeline Success Rate | 0% (cross-env) | 100% |
| VRAM (per GPU) | 6.94 GB | 3.98 GB (-43%) |
| DTG Classes | 3 | 6 |
| Localization Eval | Qualitative | IoU framework |

## 📝 Citation

```bibtex
@inproceedings{fakeshield2025,
  title={FakeShield: Explainable Image Forgery Detection and Localization with Multimodal Large Language Models},
  author={Xu, Zhipei and others},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

## ⚠️ Notes

- Model weights and datasets are excluded from this repository (see `.gitignore`)
- INT4 quantization requires >40GB GPU memory for loading
- Extended DTG was fine-tuned on a small subset; more data needed for production use
