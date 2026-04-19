# Offroad Perception Engine (OPE)
## Enterprise-Scale Semantic Segmentation Ecosystem for Unstructured Terrains

![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/Project_Status-Production_Ready-success.svg?style=for-the-badge)
![Deployment](https://img.shields.io/badge/Deployment-Edge_Optimized-blueviolet.svg?style=for-the-badge)

### 🌐 Overview
The **Offroad Perception Engine (OPE)** is a unified ecosystem designed to solve the complex challenge of semantic scene understanding in non-Euclidean, unstructured off-road environments. Originally developed for the **Hacker's Unity AI Hackathon**, OPE represents a rigorous cross-disciplinary effort to provide autonomous systems with high-fidelity spatial awareness.

Unlike standard urban road segmentation, OPE addresses the high-variance textures of vegetation, rock formations, and erratic terrain geometry through a **Model Zoo of 10 specialized neural architectures**.

---

## 🏗️ System Architecture & Workflow

The OPE pipeline follows a modular "Unified Perception Lifecycle" designed for robustness and reproducibility:

1.  **Data Normalization Layer**: Maps sparse label values `{0, 100, ..., 10000}` into a compact 10-class semantic space.
2.  **Infrastructure Optimization**: Implements inverse-frequency class weighting to mitigate severe imbalances in off-road datasets (e.g., rare 'Logs' vs. dominant 'Ground').
3.  **Multi-Model Perception Engine**: A concurrent model zoo spanning from lightweight real-time ENet to heavy-duty Transformers.
4.  **Unified Analytics & Reporting**: Automated document generation (`python-docx`) providing deep-dive metrics and failure analysis for each deployment.

---

## 🧬 The Model Zoo (Perception Research Team)

Our research team evaluated 10 distinct architectural paradigms to establish a baseline for off-road autonomy.

| Architecture | Backbone | Best IoU | Design Intent |
| :--- | :--- | :--- | :--- |
| **MobileDeepLab** | MobileNetV2 | **0.5919** | **State-of-the-art Accuracy/Efficiency** |
| **U-Net** | Symmetric Encoder | **0.5118** | High-fidelity spatial boundary preservation |
| **DINOv2 + ConvNeXt** | ViT-L Foundation | **0.4475** | Exploiting large-scale self-supervised features |
| **SegNet** | VGG-16 | 0.4370 | Memory-constrained pooling index upsampling |
| **UNet_MobileNetV2** | MobileNetV2 | 0.4201 | Real-time edge inference (<50ms) |
| **DeepLabV3+** | MobileNetV3 | 0.3858 | Multi-scale context via ASPP modules |
| **Custom_CNN** | Bespoke | 0.3659 | Terrestrial environment-tuned baseline |
| **SegFormer** | MiT-B0 | 0.3515 | Hierarchical Visual Transformer (Transformer-only) |
| **FCN-8s** | VGG-16 | 0.3119 | Multi-scale skip fusion for dense prediction |
| **ENet** | Lightweight | 0.2877 | Ultra-low latency bottleneck architecture |

---

## 🛠️ Specialized Module Breakdown

### 🔬 Perception & Research Module
- **Foundation Model Integration**: Integration of DINOv2 features for robust zero-shot style generalization.
- **Transformer Scaling**: Implementation of SegFormer architectures to leverage global context in wide-open terrains.
- **Backbone Diversity**: Native support for MobileNet, VGG, and Custom-bottleneck encoders.

### 💾 Data Engineering & Infrastructure
- **Label Remapping Engine**: Automated remapping of sparse simulation labels to compact indices.
- **Imbalance Mitigation**: Dynamic inverse-frequency weighting integrated directly into the Loss Function.
- **Augmentation Pipeline**: Specialized terrestrial augmentations including horizontal flipping and ImageNet-standardized normalization.

### ⚡ Optimization & Deployment Module
- **Edge Compliance**: Optimization for CPU-only and memory-constrained environments.
- **Checkpoint Management**: Standardized `.pth` serialization for rapid model switching.
- **Inference Suites**: Unified testing scripts for per-class metric extraction and confusion matrix generation.

---

## 🚀 Lifecycle Management (Setup & Usage)

### Professional Environment Setup
OPE requires a standardized Python 3.10+ environment.
```bash
# Initialize ecosystem
git clone https://github.com/keshavmishra27/Offroad_Autonomy_Segmentation_Model.git
cd Offroad_Autonomy_Segmentation_Model

# Install core dependencies
pip install -r requirements.txt
```

### Executing Model Inference
All models are modularized within the `Duality_Submission_Package`.
```bash
# Example: Deploying the State-of-the-Art MobileDeepLab model
cd "duality project/Offroad_Segmentation_Scripts/Duality_Submission/Duality_Submission_Package/MobileDeepLab"
python test_segmentation.py
```

---

## 👥 Professional Team Structure (The Ikigai Unified Force)

This project was built through the collaboration of several specialized "virtual" teams:
*   **Perception Team**: Focused on SOTA model architectures and IoU optimization.
*   **Infrastructure Team**: Managed dataset pipelines, label remapping, and normalization.
*   **Report & BI Team**: Developed the automated `ReportGenerator` for comprehensive stakeholder documentation.
*   **Optimization Team**: Tuned hyperparameters and class weights for terrestrial performance.

---

## 🗺️ Product Roadmap
- [ ] **Phase 1**: Integration of Attention Gates to U-Net skip connections for targeted spatial focus.
- [ ] **Phase 2**: Quantization (INT8/FP16) for deployment on mobile NPUs.
- [ ] **Phase 3**: Multi-modal sensor fusion integrating LiDAR point clouds with semantic masks.
- [ ] **Phase 4**: Temporal consistency checks in video-stream inference.

---

## ⚖️ License & Governance
Distributed under the MIT License. See `LICENSE` for more information.

**Perception Lead**: Keshav Mishra
**Organization**: Team Ikigai
**Hackathon**: Hacker's Unity AI Hackathon

---
*Unified Perception. Robust Autonomy. Offroad Excellence.*
