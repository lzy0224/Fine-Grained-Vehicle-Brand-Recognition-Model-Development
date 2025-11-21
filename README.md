# Fine-Grained-Vehicle-Brand-Recognition-Model-Development

## 📖 Overview

This project implements a robust **two-stage vehicle recognition system** capable of detecting vehicle presence and performing fine-grained classification for **10 specific vehicle brands** (e.g., Audi A4, VW Magotan, Toyota Corolla).

Built upon the **ResNet50** architecture with a **Two-Phase Transfer Learning** strategy, the model achieves a top-tier accuracy of **98.54%** on the designated test dataset, significantly surpassing the project baseline of 80%.

## 📥 Resource Download

Due to GitHub file size limitations, the trained model weights and the dataset are hosted on Baidu Netdisk. Please download them before running the code.

| File | Download Link | Extraction Code | Description |
| :--- | :--- | :--- | :--- |
| **Trained Model** | [Click to Download](https://pan.baidu.com/s/1k578SnrsoAHn6H-oEL_9pA) | **1234** | `best_resnet_finetuned.h5`. Place this file in the project root directory. |
| **Dataset** | [Click to Download](https://pan.baidu.com/s/1GnQ0aUciBN1_x85Qn-swWg) | **3rms** | Unzip the content into an `image/` folder in the project root directory. |

## 🚀 Key Features

- **Two-Level Recognition**: 
  - **Level 1**: Vehicle Existence Detection (based on Softmax confidence thresholding).
  - **Level 2**: Fine-Grained Brand Classification (10 Classes).
- **Advanced Architecture**: 
  - Backbone: **ResNet50** (pre-trained on ImageNet).
  - Head: Global Average Pooling + Dropout + Dense Layer with L2 Regularization.
- **Training Strategy**: 
  - **Phase 1 (Warm-up)**: Frozen backbone, high learning rate for the classifier head.
  - **Phase 2 (Fine-tuning)**: Unfrozen backbone, low learning rate (`1e-5`) for deep feature adaptation.
- **Rigorous Pipeline**: Standardized `caffe` style pre-processing and strict train/test data isolation.

## 📊 Experimental Results

Performance evaluated on the strictly isolated test set (`re_id_1000_test.txt`, 5,000 images):

| Metric | Value |
| :--- | :--- |
| **Test Accuracy** | **98.54%** |
| **Test Loss** | 0.1094 |
| **Detection Logic** | Threshold-based (Confidence > 0.65) |

> **Note**: The model demonstrates exceptional capability in distinguishing visually similar car models within the provided dataset context.

## ⚠️ Critical Analysis & Limitations

While the model achieves near-perfect scores on the assignment dataset, an honest technical analysis reveals specific constraints regarding real-world generalization:

1.  **Dataset Bias (Re-ID Nature)**: The provided dataset has characteristics of **Vehicle Re-Identification** data (consecutive shots of the same vehicle instance). The model may partially rely on instance-specific features (e.g., custom stickers, specific rims) rather than general class features to achieve high scores.
2.  **Spurious Correlations**: Due to fixed camera angles in the training set, the model might learn correlations between background textures (e.g., specific road tiles) and vehicle labels (Shortcut Learning).
3.  **Domain Gap**: Tests on external, web-crawled street view images show a drop in confidence, highlighting the **Sim-to-Real gap**. For production deployment, **Domain Adaptation** techniques would be required.

## 🛠️ Installation

1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 📂 Data Preparation

Ensure your project directory is structured as follows. **Note**: The image dataset is not included in this repo due to size constraints.

```text
project_root/
├── image/                  # Dataset images (Source)
├── re_id_1000_train.txt    # Training index file
├── re_id_1000_test.txt     # Testing index file
├── main.py                 # Main training script
└── requirements.txt        # Dependencies
