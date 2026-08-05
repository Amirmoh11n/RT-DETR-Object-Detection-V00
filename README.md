# Transfer Learning with RT-DETR for Object Detection on Pascal VOC 2012

## Overview

This repository demonstrates how to apply **transfer learning** with **RT-DETR (Real-Time Detection Transformer)** for object detection on the **Pascal VOC 2012** dataset.

Rather than training a detector from scratch, a pretrained RT-DETR model is fine-tuned to recognize the 20 object classes in Pascal VOC. The project provides a complete and modular deep learning pipeline covering dataset preparation, training, evaluation, and inference using PyTorch.

---

## Features

- Transfer learning with a pretrained RT-DETR model
- Fine-tuning on the Pascal VOC 2012 dataset
- End-to-end object detection pipeline
- Modular PyTorch project structure
- Training, validation, and inference workflows
- Easy-to-extend architecture for custom datasets
- Clean code organization following software engineering best practices

---

## Model

### RT-DETR (Real-Time Detection Transformer)

RT-DETR is a transformer-based object detection architecture that combines the strong representation capabilities of Vision Transformers with efficient real-time inference.

In this project, pretrained RT-DETR weights are used as the starting point for **transfer learning**, allowing the model to adapt quickly to the Pascal VOC dataset while reducing training time and improving convergence.

---

## Transfer Learning Pipeline

The training process follows a standard transfer learning workflow:

1. Load pretrained RT-DETR weights
2. Prepare the Pascal VOC 2012 dataset
3. Preprocess images and annotations
4. Fine-tune the detector on Pascal VOC
5. Evaluate the adapted model
6. Perform inference on unseen images

Using pretrained weights enables the model to leverage previously learned visual representations instead of learning all features from scratch.

---

## Dataset

### Pascal VOC 2012

The project uses the **Pascal VOC 2012** object detection dataset, which contains **20 object categories**, including:

- Person
- Car
- Bus
- Bicycle
- Train
- Dog
- Cat
- Horse
- Chair
- Bottle

and other common object classes.

---

## Project Goals

- Apply transfer learning using a pretrained RT-DETR model
- Explore transformer-based object detection
- Build a modular object detection pipeline with PyTorch
- Separate data processing, model definition, training, evaluation, and inference
- Develop a reusable codebase for future object detection projects
- Gain practical experience with Vision Transformer architectures

---

## Project Structure

```text
RTDETR-Object-Detection/
│
├── configs/
├── data/
│   └── dataset.py
├── models/
│   └── rtdetr.py
├── training/
│   └── train.py
├── evaluation/
│   ├── evaluate.py
│   └── metrics.py
├── inference/
│   ├── predict.py
│   └── outputs/
├── checkpoints/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Workflow

```text
Pretrained RT-DETR
        │
        ▼
 Pascal VOC 2012
        │
        ▼
 Data Preparation
        │
        ▼
 Transfer Learning
        │
        ▼
 Fine-Tuning
        │
        ▼
 Model Evaluation
        │
        ▼
 Image Inference
```

---

## Results

After training, the project produces:

- Fine-tuned model checkpoints
- Trained model weights
- Evaluation metrics
- Object detection predictions on custom images

These outputs can be used for further evaluation, deployment, or continued fine-tuning on other datasets.

---

## Current Status

### Completed

- Pascal VOC dataset preparation
- RT-DETR integration
- Transfer learning pipeline
- Fine-tuning workflow
- Model training
- Image inference
- Modular project structure
- Migration from Google Colab to a local development environment

### Future Improvements

- mAP evaluation and benchmarking
- Video inference pipeline
- Real-time webcam object detection
- ONNX and TensorRT export
- Mixed-precision training
- Hyperparameter optimization

---

## Learning Outcomes

This project provided practical experience with:

- Transfer learning
- Fine-tuning large Vision Transformer models
- RT-DETR architecture
- Transformer-based object detection
- Pascal VOC dataset processing
- PyTorch training workflows
- Modular deep learning software design
- Computer vision model deployment workflows

---

## Requirements

- Python 3.10+
- PyTorch
- Transformers
- Torchvision
- OpenCV
- NumPy

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Acknowledgments

- RT-DETR Authors
- Pascal VOC Dataset Contributors
- Hugging Face
- PyTorch Community
