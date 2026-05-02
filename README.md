# Brain Tumor Classification from MRI Scans Using Convolutional Neural Networks

**Student:** Divyansh Yecho
**Roll Number:** 20231086
**Course:** Image and Video Processing
**Date:** May 2026

---

## Kaggle Notebook

The full training notebook with all outputs, plots, and results is available here:

[View Kaggle Notebook](https://www.kaggle.com/code/divyanshyecho/project-divyansh-yecho)

---

## Project Overview

Brain tumors are among the most life-threatening medical conditions, and early accurate diagnosis is critical to patient survival. Currently, diagnosis requires a trained radiologist or neurosurgeon to manually examine MRI scans — a process that is time-consuming, expensive, and highly dependent on human expertise. Even experienced radiologists can struggle to distinguish between tumor types in early stages, because the visual differences between certain tumor categories are subtle and require years of specialised training to recognise reliably.

This project builds an automated deep learning system that classifies brain MRI scans into one of four categories using a fine-tuned ResNet50 convolutional neural network trained via transfer learning. The system outputs not just a predicted class label, but also a full probability distribution across all four categories, allowing the user to assess the model's confidence and uncertainty for each prediction.

---

## Problem Statement

### Task
Automatically classify brain MRI scans into one of four categories:

| Class | Diagnosis |
|-------|-----------|
| 0 | Glioma |
| 1 | Meningioma |
| 2 | No Tumor |
| 3 | Pituitary Tumor |

### Why This Is Hard
Each tumor type exhibits distinct but visually similar structural patterns in MRI imagery. Glioma and Meningioma in particular can appear superficially similar in shape and intensity. Detecting these subtle differences requires hierarchical feature extraction across multiple spatial scales — which is exactly what deep CNNs are designed to do. The problem cannot be solved by simple thresholding or hand-crafted feature methods and demands learned representations from data.

---

## Dataset

**Name:** Brain Tumor MRI Dataset
**Author:** Masoud Nickparvar
**Source:** [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

| Split | Images per Class | Total |
|-------|-----------------|-------|
| Training | 1,400 | 5,600 |
| Testing | 400 | 1,600 |
| **Total** | **1,800** | **7,200** |

The dataset is perfectly balanced across all four classes, eliminating the class imbalance problem common in medical imaging datasets. Images are pre-labelled and organised into class-named folders for both Training and Testing splits, eliminating the need for manual annotation or train/test splitting.

---

## Model Architecture

The model is a modified ResNet50 with a custom classification head, trained via transfer learning.

### Base: ResNet50

ResNet50 is a 50-layer deep residual network originally trained on ImageNet (1.2 million images, 1000 classes). It consists of:

- An initial 7×7 convolution layer with 64 filters and stride 2
- A max pooling layer
- Four residual layer groups (`layer1` through `layer4`), each containing multiple bottleneck blocks
- Each bottleneck block has three convolutions: 1×1 → 3×3 → 1×1, with a skip connection that adds the input directly to the output

The skip connections (residual connections) are the key innovation of ResNet — they allow gradients to flow directly through the network during backpropagation, making very deep networks trainable without vanishing gradients.

### Transfer Learning Strategy: Partial Fine-Tuning

Rather than training from scratch, the model starts with ImageNet pretrained weights. The backbone is then partially frozen:

- `layer1`, `layer2`, `layer3` — **frozen** (weights fixed, not updated during training)
- `layer4` — **unfrozen** (fine-tuned on the MRI data)
- `fc` (classification head) — **replaced and trained from scratch**

This strategy is deliberately chosen for medical imaging. The early layers of ResNet50 detect universal low-level features (edges, textures, gradients) that transfer well from ImageNet. The later layers detect higher-level, domain-specific features — by unfreezing `layer4`, the model can adapt these higher-level detectors to the specific visual characteristics of brain MRI scans, rather than being locked into ImageNet-specific features.

### Custom Classification Head

The original ResNet50 `fc` layer (which outputs 1000 ImageNet classes) is replaced with:

```
Linear(2048 → 256)
ReLU
Dropout(p=0.4)
Linear(256 → 4)
```

The 40% dropout between the two linear layers prevents overfitting — during training, 40% of neurons are randomly disabled each forward pass, forcing the network to learn redundant representations that generalise better to unseen data.

### Full Architecture Summary

```
Input: (batch, 3, 224, 224)
    ↓
Conv1: 7×7, stride 2, 64 filters        [frozen]
    ↓
MaxPool: 3×3, stride 2                   [frozen]
    ↓
Layer1: 3× Bottleneck blocks, 256ch      [frozen]
    ↓
Layer2: 4× Bottleneck blocks, 512ch      [frozen]
    ↓
Layer3: 6× Bottleneck blocks, 1024ch     [frozen]
    ↓
Layer4: 3× Bottleneck blocks, 2048ch     [FINE-TUNED]
    ↓
AdaptiveAvgPool: (1, 1) → 2048-dim vector
    ↓
Linear(2048 → 256) + ReLU + Dropout(0.4) [trained]
    ↓
Linear(256 → 4)                           [trained]
    ↓
Output: (batch, 4) logits → Softmax → class probabilities
```

**Total trainable parameters:** ~13.5 million (layer4 + custom head)
**Frozen parameters:** ~23.5 million (layers 1–3 + initial conv)

---

## Image Preprocessing Pipeline

MRI scans vary in resolution, contrast, and brightness across different scanners and acquisition settings. A standardised preprocessing pipeline is applied before any image enters the model.

### Training Transforms (with augmentation)
```
Resize to 224×224
RandomHorizontalFlip          (p=0.5)
RandomRotation(±15°)
ColorJitter(brightness=0.2, contrast=0.2)
ToTensor
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

The normalisation values are the ImageNet channel statistics. Although MRI scans are grayscale, they are converted to 3-channel RGB to be compatible with the pretrained ResNet50 backbone.

**Augmentation choices justified:**
- Horizontal flip — valid because the brain has left-right symmetry in MRI
- Rotation ±15° — mimics realistic variation in patient positioning
- Color jitter — simulates variation in MRI scanner contrast settings
- Vertical flip is intentionally excluded — flipping a brain upside down creates an anatomically impossible image

### Evaluation Transforms (no augmentation)
```
Resize to 224×224
ToTensor
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

---

## Training Configuration

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Batch Size | 32 |
| Epochs | 25 |
| Dropout | 0.4 |
| Validation Split | 15% of training data |
| LR Scheduler | ReduceLROnPlateau (patience=3, factor=0.5) |
| Loss Function | CrossEntropyLoss |
| GPU | NVIDIA T4 (Kaggle) |

The learning rate of 1e-4 (rather than the more common 1e-3) is deliberately chosen because `layer4` contains pretrained weights — a higher learning rate would destroy the learned features. The ReduceLROnPlateau scheduler halves the learning rate when validation loss stops improving for 3 consecutive epochs, allowing fine convergence in later training.

---

## Results

| Metric | Score |
|--------|-------|
| Test Accuracy | See Kaggle notebook |
| Best Validation Accuracy | See Kaggle notebook |
| Weighted Precision | See Kaggle notebook |
| Weighted Recall | See Kaggle notebook |
| Weighted F1 Score | See Kaggle notebook |
| Total Test Images | 1,600 |

All detailed results including confusion matrix, per-class accuracy, training curves, sample predictions, misclassified sample analysis, and per-class probability breakdowns are available in the Kaggle notebook linked above.

---

## Repository Structure

```
project_divyansh_yecho/
│
├── checkpoints/
│   └── final_weights.pth       # Saved model weights (best validation accuracy)
│
├── data/
│   ├── glioma_01.jpg            # 10 sample images per class
│   ├── glioma_02.jpg
│   ├── ...
│   ├── meningioma_01.jpg
│   ├── ...
│   ├── notumor_01.jpg
│   ├── ...
│   └── pituitary_10.jpg        # 40 images total
│
├── results/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── sample_predictions.png
│   ├── misclassified_samples.png
│   ├── per_class_accuracy.png
│   ├── confidence_distribution.png
│   ├── probability_breakdown.png
│   └── final_summary.png
│
├── config.py                   # All hyperparameters and paths in one place
├── dataset.py                  # BrainTumorDataset class and dataloader
├── model.py                    # BrainTumorClassifier (ResNet50 + custom head)
├── train.py                    # Training loop function
├── predict.py                  # Inference function for single images or batches
└── interface.py                # Standardised exports for grader
```

---

## File Descriptions

**`config.py`** — Single source of truth for all hyperparameters: paths, class names, image dimensions, batch size, learning rate, dropout, validation split, ImageNet normalisation statistics, and device. All other files import from here.

**`dataset.py`** — Contains the `BrainTumorDataset` PyTorch Dataset class, train/eval transform pipelines, and the `build_dataloaders()` function that constructs the train, validation, and test DataLoaders from the Kaggle dataset directory.

**`model.py`** — Defines `BrainTumorClassifier`, which loads the pretrained ResNet50 backbone, freezes layers 1–3, unfreezes layer4, and replaces the final fully connected layer with the custom classification head.

**`train.py`** — Contains `train_brain_tumor_model()`, which runs the full training loop with epoch-level logging, ReduceLROnPlateau scheduling, and best-model checkpointing based on validation accuracy.

**`predict.py`** — Contains `classify_mri_scans(list_of_img_paths)`, which loads the saved weights and returns the predicted class, confidence percentage, and full 4-class probability breakdown for each image path provided.

**`interface.py`** — Re-exports all key components under standardised names for programmatic evaluation: `TheModel`, `the_trainer`, `the_predictor`, `TheDataset`, `the_dataloader`, `the_batch_size`, `total_epochs`.

---

## How to Run Inference

```python
from predict import classify_mri_scans

results = classify_mri_scans([
    'data/glioma_01.jpg',
    'data/meningioma_03.jpg',
    'data/notumor_07.jpg'
])

for r in results:
    print(r['predicted_class'], r['confidence'], r['probabilities'])
```

---

## Goals Achievement

| Goal | Status |
|------|--------|
| Train ResNet50 via transfer learning on 4-class MRI dataset | Done |
| Achieve ≥90% test accuracy | See notebook |
| Document preprocessing pipeline (resize, normalise, augment) | Done |
| Confusion matrix showing per-class performance | Done |
| Per-class accuracy bar chart + classification report | Done |
| Training and validation accuracy/loss curves | Done |
| Predictions on ≥16 unseen test images with confidence scores | Done |
| Probability breakdown for one sample per class | Done |
| Misclassified sample analysis | Done |
| Stretch goal: upload interface | Not attempted |
