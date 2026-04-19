# Offroad Autonomy: Semantic Segmentation Model Zoo
## Hacker's Unity AI Hackathon Submission

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![Models](https://img.shields.io/badge/Models-10-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

This repository contains a comprehensive suite of **10 semantic segmentation projects** developed for the Hacker's Unity AI Hackathon. The goal is to enable robust off-road autonomy by accurately segmenting challenging environmental features such as vegetation, rocks, sky, and obstacles in unstructured terrain.

---

##  Model Leaderboard

We implemented and evaluated 10 different architectures to find the optimal balance between accuracy (IoU) and real-time performance.

| Rank | Model Architecture | Best Validation IoU | Backbone / Features |
| :--- | :--- | :--- | :--- |
| 🥇 | **MobileDeepLab** | **0.5919** | Optimized MobileNet |
| 🥈 | **U-Net** | **0.5118** | Classic Encoder-Decoder |
| 🥉 | **DINOv2 + ConvNeXt** | **0.4475** | Foundation Model + Transformer |
| 4 | **SegNet** | 0.4370 | VGG-based Encoder-Decoder |
| 5 | **UNet_MobileNetV2** | 0.4201 | Efficiency Optimized |
| 6 | **DeepLabV3+** | 0.3858 | MobileNetV3 Backbone |
| 7 | **Custom_CNN** | 0.3659 | Baseline Architecture |
| 8 | **SegFormer (MiT-B0)** | 0.3515 | Hierarchical Transformer |
| 9 | **FCN** | 0.3119 | Fully Convolutional |
| 10 | **ENet** | 0.2877 | Real-time Optimized |

---

##  Project Structure

The submission is organized into independent projects, each containing its own training, testing, and visualization pipeline.

```text
Duality_Submission_Package/
├── MobileDeepLab/           # SOTA Performance
├── U-Net/                   # Robust Baseline
├── DINOv2_ConvNeXt/         # Modern Transformer Approach
├── DeepLabV3Plus_MobileNetV3/
├── UNet_MobileNetV2/
├── SegNet/
├── Custom_CNN/
├── SegFormer_MiT_B0/
├── FCN/
└── ENet/                    # Lightweight / Real-time
```

---

##  Getting Started

### Prerequisites
- Python 3.10+
- PyTorch (compatible with your CUDA version)
- `pip install -r requirements.txt` (available in each project folder)

### Usage
Each model folder is self-contained. To evaluate any model:
1. Navigate to the model directory.
2. Run `python test_segmentation[X].py` (replace X with project number).
3. View results in the output folder specified in the script.

---

## 📄 Documentation

For a deep dive into the methodology, training strategy, and failure analysis, please refer to:
- [Hacker's Unity AI Hackathon - Segmentation Documentation.pdf](./duality%20project/Hacker's%20Unity%20AI%20Hackathon%20-%20Segmentation%20Documentation.pdf)
- Individual `README.md` files within each project directory.

---

##  Key Features
- **Diverse Architectures**: From standard CNNs to cutting-edge Vision Transformers (ViT).
- **Efficiency Focus**: Several models (MobileDeepLab, ENet) are optimized for edge deployment.
- **Robust Evaluation**: Comprehensive metrics including per-class IoU, Dice coefficients, and confusion matrices.
- **Ready for Deployment**: Saved `.pth` checkpoints and standardized inference scripts.

---
*Developed for the Hacker's Unity AI Hackathon - Offroad Autonomy Challenge.*