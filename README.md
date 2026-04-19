# Multi-Scale Contrastive Framework for Zero-Shot Deepfake Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Author:** Triambak Ragavan | Vellore Institute of Technology (VIT), Chennai
> 
> This repository contains the official implementation of the research paper: *"Multi-Scale Contrastive Frameworks for Zero-Shot Deepfake Detection."*

## Abstract
The rapid evolution of Generative AI has made robust deepfake detection critical. Most modern detectors suffer from **"shortcut learning"**—memorizing generator-specific high-frequency fingerprints rather than learning actual structural anomalies. As a result, their performance collapses when exposed to novel, unseen AI generators. 

This project introduces a unique Multi-Scale Contrastive Framework that forces the model to focus on macro and micro structural inconsistencies. By utilizing aggressive augmentations, an explicit multi-scale image pyramid, and unsupervised MoCo v2 weights on a ResNet-50 backbone, this model achieves **state-of-the-art zero-shot generalization** across entirely unseen generative architectures (Latent Diffusion, DDPM, StyleGAN3, etc.).

---

##  Architecture: The 3-Phase Pipeline

The model pipeline is designed to explicitly destroy pixel-level noise and extract robust structural features.

1. **Phase 1: Aggressive Augmentation**
   * Heavily augments input images using Coarse Dropout (spatial cutout), Gaussian Blurring, and JPEG compression.
   * **Goal:** Systematically destroy high-frequency generator fingerprints and prevent shortcut learning.
2. **Phase 2: Explicit Multi-Scale Pyramid & Feature Extraction**
   * Generates a physical image pyramid at 3 scales: $1x$ (Original), $0.5x$ (Half Res), and $0.25x$ (Quarter Res).
   * Passes these scales through a **frozen ResNet-50 backbone** initialized with **MoCo v2 weights** (trained via unsupervised contrastive learning for superior texture/semantic mapping).
   * Concatenates the features into a unified vector.
3. **Phase 3: Feature Fusion & End-to-End Fine-Tuning**
   * Features are fed into a custom MLP classification head. 
   * The backbone is initially frozen to prevent overwriting contrastive representations. 
   * Finally, the entire network undergoes a micro-learning rate fine-tuning phase ($\alpha=1\times10^{-5}$) using AdamW.

---

## Datasets & Evaluation

* **Training Set:** Synthetic images generated primarily by modern diffusion models (Stable Diffusion v1.5/v2, Midjourney).
* **Zero-Shot Evaluation Set:** Tested on the open-set **ArtiFact Dataset**.
* **Unseen Generators Tested:** StyleGAN3, Latent Diffusion, Projected GAN, DDPM, StarGAN, VQ-Diffusion, and Taming Transformer.

### Key Results
Our Multi-Scale MoCo v2 model significantly outperformed the standard ResNet-50 baseline in zero-shot cross-dataset generalization tests:
* **Latent Diffusion:** Recall improved from 47.2% to **79.2%** (+32.0%).
* **DDPM:** Recall improved from 12.6% to **66.4%** (+53.8%).
* Consistently achieved **98%+ Precision** across most novel generators.

---

## Explainable AI (XAI) Validation
To prove the model overcame shortcut learning, **Integrated Gradients (IG)** were used for visual explainability. While the baseline ResNet-50 fixated on random background noise (fingerprints), XAI heatmaps confirm that this Multi-Scale architecture correctly targets structural features (e.g., facial asymmetry, irregular blending boundaries).

---

##  Quick Start (Demo)

You can run the full training pipeline and inference script in a cloud environment:
👉 https://www.kaggle.com/code/triambak23bai1429/deepfake-detection

### Local Installation
```bash
# Clone the repository
git clone [https://github.com/yourusername/multi-scale-deepfake-detector.git](https://github.com/yourusername/multi-scale-deepfake-detector.git)
cd multi-scale-deepfake-detector

# Install dependencies
pip install -r requirements.txt
