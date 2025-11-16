# EM-SliceCrafter: Distilling Lightweight 2D CNNs for Efficient 3D EM Neuron Segmentation

## 📢 About This Work

This repository contains the official implementation of our paper submitted to **IEEE Transactions on Medical Imaging (TMI)**. This work is an **extension version** of our CVPR-2024 paper "[Cross-dimension Affinity Distillation for 3D EM Neuron Segmentation](https://github.com/liuxy1103/CAD)".

**Authors:** Xiaoyu Liu, Haoyuan Shi, Yinda Chen, Miaomiao Cai, Xuejin Chen, Zhiwei Xiong  
**Affiliation:** University of Science and Technology of China, MoE Key Laboratory of Brain-inspired Intelligent Perception and Cognition

---

## 🔥 Highlights

- **⚡ 33× Faster Inference:** Achieves state-of-the-art accuracy with only 1/33 inference time compared to 3D CNNs
- **🎯 Dual-Distillation Framework:** Novel combination of Cross-Dimension Affinity Distillation (CAD) and Internal-Dimension Affinity Distillation (IAD)
- **🤖 SAM Integration:** First work to leverage Segment Anything Model (SAM) for EM neuron segmentation via learnable adapter
- **🏆 SOTA Performance:** Superior results on CREMI, AC3/4, Wafer4, and ZEBRAFINCH datasets
- **💾 Lightweight Design:** Only 0.9M parameters vs. 1.5M-84M for competing 3D CNNs
- **🌍 Robust Generalization:** Validated across diverse EM modalities (ssTEM, mbSEM, SBEM) and biological samples

---

## 📖 Abstract

Accurate 3D neuron segmentation from large-scale electron microscopy (EM) volumes is a cornerstone of connectomics, yet it is hampered by a trade-off between accuracy and computational efficiency. While 3D Convolutional Neural Networks (CNNs) offer high accuracy by predicting 3D affinity maps, their expensive computational cost and limited input size impede their application on large-scale data.

To overcome this, we introduce **EM-SliceCrafter**, a novel framework that employs a lightweight 2D CNN for both efficient and precise 3D neuron segmentation. Our method generates a 3D affinity map by computing distances between embedding maps of adjacent 2D slices. To enrich the 2D network's contextual understanding, we propose a dual-distillation strategy:

1. **Cross-Dimension Affinity Distillation (CAD):** Transfers inter-slice structural knowledge from a 3D teacher network, ensuring 3D continuity
2. **Internal-Dimension Affinity Distillation (IAD):** Leverages a powerful teacher built on the pre-trained Segment Anything Model (SAM) encoder, using a learnable adapter, to distill fine-grained intra-slice boundary details

Furthermore, a **Feature Grafting Interaction (FGI)** module deepens knowledge transfer by integrating embeddings across the student and both teacher networks. Extensive experiments on diverse EM datasets, featuring various imaging modalities and resolutions, confirm that EM-SliceCrafter surpasses state-of-the-art methods while achieving a significant **1/33 reduction in inference time**.

---

## 🆕 What's New in This Extension

Compared to our CVPR-2024 work [CAD](https://github.com/liuxy1103/CAD), this TMI submission provides substantial extensions:

### Major Contributions

- **🎨 Internal-Dimension Affinity Distillation (IAD):** A new distillation strategy complementing CAD. While CAD handles inter-slice dependencies, IAD sharpens intra-slice details using a strong 2D teacher built on the pre-trained SAM encoder
- **🔗 Enhanced Feature Grafting Interaction (FGI):** Upgraded to serve both CAD and IAD distillation paths, deepening knowledge transfer through dynamic feature integration
- **📊 Expanded Validation:** Comprehensive evaluation across datasets from different EM modalities (ssTEM, mbSEM, SBEM) and various biological samples (Drosophila, mouse cortex, zebra finch)
- **🏅 Superior Performance:** Enhanced framework demonstrates better segmentation accuracy across all benchmarks while maintaining high computational efficiency
- **⚡ Maintained Efficiency:** Achieves superior accuracy improvements while preserving the 2D-based approach's efficiency advantage (up to 33× speedup)

---

## 🔧 Environment Setup

You can get the environment through:

```bash
pip install -r requirements.txt
```

## 🚀 Getting Started
### Training
```bash
cd scripts_2_5d_3d
python3 main_general_dual_SAM_3d_v8.py \
  --cfg=2d_SAM_MEC_slice1.0_cross1.0_interaction0.1_Lsam0.1_L3d1.0_interation0.1 \
  --cfg_3d=3d_MEC_h160_noft_lr_ratio1_ft10000
```

### Inference
```bash
cd scripts_2_5d_3d
python inference_affs.py \
  --cfg=2d_SAM_MEC_slice1.0_cross1.0_interaction0.1_Lsam0.1_L3d1.0_interation0.1 \
  --model_name=wafer4 \
  --mode=wafer4 \
  --test_split=25
```
