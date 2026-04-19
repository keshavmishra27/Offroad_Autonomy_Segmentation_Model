# 🏜️ Offroad Autonomy: Semantic Segmentation Model Zoo
### Code Royale: The Final Iteration

This repository contains **10 independent semantic segmentation projects** submitted to the **Code Royale: The Final Iteration**. Each project was developed by a separate team and represents a distinct architectural approach to solving off-road scene understanding for Unmanned Ground Vehicles (UGVs).

The dataset and problem statement were provided by **Duality AI** using their Falcon simulation platform — all training data is high-quality synthetic imagery of desert environments, annotated across 10 semantic classes.

---

## 🏆 Leaderboard

| Rank | Team Name | Model Architecture | Best Val IoU | Backbone / Features |
|------|-----------|-------------------|-------------|---------------------|
| 🥇 1 | **Team Apex** | MobileDeepLab | **0.5919** | Optimized MobileNetV2 |
| 🥈 2 | **Team NeuralNomads** | U-Net | 0.5118 | Classic Encoder-Decoder |
| 🥉 3 | **Team Horizon** | DINOv2 + ConvNeXt | 0.4475 | Foundation Model + Transformer |
| 4 | **Team GridIron** | SegNet | 0.4370 | VGG-based Encoder-Decoder |
| 5 | **Team SwiftSeg** | UNet + MobileNetV2 | 0.4201 | Efficiency Optimized |
| 6 | **Team DeepDrift** | DeepLabV3+ | 0.3858 | MobileNetV3 Backbone |
| 7 | **Team PixelForge** | Custom CNN | 0.3659 | Baseline Architecture |
| 8 | **Team Transformers** | SegFormer (MiT-B0) | 0.3515 | Hierarchical Transformer |
| 9 | **Team RoadRunner** | FCN | 0.3119 | Fully Convolutional |
| 10 | **Team EdgeCore** | ENet | 0.2877 | Real-time Optimized |

> IoU scores are computed on the held-out validation set provided by Duality AI. Test set evaluation was conducted on unseen desert environment images from a separate Falcon simulation location.

---

## 🎯 Challenge Overview

**Organizer:** Hacker's Unity AI Hackathon × Duality AI  
**Challenge:** Code Royale: The Final Iteration  
**Track:** Offroad Autonomy — Semantic Scene Segmentation  
**Evaluation Metric:** Mean IoU (Intersection over Union) — 80 points  
**Report Quality:** Structured findings & documentation — 20 points  

### The Problem
Unmanned Ground Vehicles navigating unstructured desert terrain need to understand their environment at the pixel level. This means classifying every pixel in a camera frame into one of 10 semantic categories in real time. Teams were tasked with training the most accurate and generalizable segmentation model possible using only synthetic data generated from Duality AI's Falcon platform.

### Dataset
All data was generated from Duality AI's **FalconEditor** using geospatial-based desert environment digital twins.

| Class ID | Class Name |
|----------|------------|
| 100 | Trees |
| 200 | Lush Bushes |
| 300 | Dry Grass |
| 500 | Dry Bushes |
| 550 | Ground Clutter |
| 600 | Flowers |
| 700 | Logs |
| 800 | Rocks |
| 7100 | Landscape (general ground) |
| 10000 | Sky |

---

## 📂 Repository Structure

<pre><code>
Duality_Submission_Package/
│
├── MobileDeepLab/                  # 🥇 Team Apex — SOTA Performance
│   ├── model.py
│   ├── train.py
│   ├── test.py
│   ├── dataset.py
│   ├── config.py
│   ├── utils.py
│   ├── benchmark.py
│   ├── report_generator.py
│   ├── requirements.txt
│   ├── README.md
│   └── runs/
│
├── U-Net/                          # 🥈 Team NeuralNomads — Robust Baseline
│   ├── model.py
│   ├── train.py
│   ├── test.py
│   ├── dataset.py
│   ├── config.py
│   ├── utils.py
│   ├── report_generator.py
│   ├── requirements.txt
│   ├── README.md
│   └── runs/
│
├── DINOv2_ConvNeXt/                # 🥉 Team Horizon — Foundation Model
│   ├── model.py
│   ├── train.py
│   ├── test.py
│   ├── dataset.py
│   ├── config.py
│   ├── utils.py
│   ├── report_generator.py
│   ├── requirements.txt
│   ├── README.md
│   └── runs/
│
├── SegNet/                         # Team GridIron
│   └── ...
│
├── UNet_MobileNetV2/               # Team SwiftSeg
│   └── ...
│
├── DeepLabV3Plus_MobileNetV3/      # Team DeepDrift
│   └── ...
│
├── Custom_CNN/                     # Team PixelForge
│   └── ...
│
├── SegFormer_MiT_B0/               # Team Transformers
│   └── ...
│
├── FCN/                            # Team RoadRunner
│   └── ...
│
└── ENet/                           # Team EdgeCore — Real-time Optimized
└── ...
---
</code></pre>
## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- PyTorch with CUDA support
- Anaconda or Miniconda (recommended)

### Setup
Each project is fully self-contained with its own dependencies.

```bash
# Clone the repository
git clone https://github.com/your-username/code-royale-segmentation.git
cd code-royale-segmentation

# Navigate to any project
cd MobileDeepLab

# Install dependencies
pip install -r requirements.txt
```

### Running Inference
```bash
# Test any trained model on unseen images
python test.py

# Outputs saved to runs/predictions/
```

### Regenerating Reports
```bash
# Auto-generate the full submission report from training outputs
python report_generator.py

# Outputs: report.docx + README.md + runs/report_figures/
```

---

## 📊 Architecture Comparison

| Model | Params | Inference Time | IoU | Best For |
|-------|--------|---------------|-----|----------|
| MobileDeepLab | ~5.8M | ~35ms | 0.5919 | Speed + Accuracy |
| U-Net | ~31M | ~48ms | 0.5118 | Accuracy |
| DINOv2 + ConvNeXt | ~87M | ~120ms | 0.4475 | Rich Features |
| SegNet | ~29M | ~42ms | 0.4370 | Memory Efficiency |
| UNet + MobileNetV2 | ~8M | ~38ms | 0.4201 | Balanced |
| DeepLabV3+ | ~11M | ~45ms | 0.3858 | Multi-scale Context |
| Custom CNN | ~4M | ~28ms | 0.3659 | Simplicity |
| SegFormer MiT-B0 | ~3.7M | ~52ms | 0.3515 | Transformer Baseline |
| FCN | ~134M | ~65ms | 0.3119 | Historical Reference |
| ENet | ~0.36M | ~18ms | 0.2877 | Edge Deployment |

> Inference times measured on NVIDIA GPU with 512×512 input. Target benchmark: <50ms per image.

---

## 🛠️ Key Features

- **10 Distinct Architectures** — from classic CNNs (FCN, U-Net, SegNet) to modern transformers (SegFormer, DINOv2)
- **Shared Dataset, Fair Comparison** — all teams trained on identical Duality AI synthetic desert data
- **Standardized Evaluation** — every project reports mean IoU, per-class IoU, confusion matrix, and inference time
- **Auto-generated Reports** — each project includes `report_generator.py` that produces a full 8-page PDF/DOCX submission report
- **Deployment Ready** — saved `.pth` checkpoints and standardized inference scripts in every folder
- **Speed vs Accuracy Tradeoff** — from ENet at 18ms to DINOv2 at 120ms, the full spectrum is covered

---

## 📄 Documentation

| Document | Description |
|----------|-------------|
| `Hacker's_Unity_AI_Hackathon_Segmentation_Documentation.pdf` | Official challenge brief and judging criteria |
| `[Project]/README.md` | Per-project setup, training, and reproduction instructions |
| `[Project]/report.docx` | Full 8-page hackathon submission report per team |
| `[Project]/runs/report_figures/` | All generated charts, confusion matrices, and failure analysis |

---

## 🔗 Resources

- 🌐 [Duality AI Falcon Platform](https://falcon.duality.ai)
- 💬 [Duality AI Discord Community](https://discord.com/invite/dualityfalconcommunity)
- 📦 [Dataset Download](https://falcon.duality.ai/secure/documentation/hackathon-segmentation-desert)

---

## ⚠️ Important Notes

- Test images were **never used during training** — strict separation maintained across all 10 projects
- All models trained **exclusively on the provided Duality AI dataset**
- Model weights (`.pth` files) may not be included in the repo due to file size — run `train.py` to reproduce
- Each project's `report_generator.py` auto-fills all metrics from `runs/log.csv` — no hardcoded numbers

---

## 👥 Teams

| Team | Project |
|------|---------|
| Team Apex | MobileDeepLab |
| Team NeuralNomads | U-Net |
| Team Horizon | DINOv2 + ConvNeXt |
| Team GridIron | SegNet |
| Team SwiftSeg | UNet + MobileNetV2 |
| Team DeepDrift | DeepLabV3+ |
| Team PixelForge | Custom CNN |
| Team Transformers | SegFormer MiT-B0 |
| Team RoadRunner | FCN |
| Team EdgeCore | ENet |

---

*Submitted to Code Royale: The Final Iteration  
*Challenge provided by Duality AI | Falcon Simulation Platform*
