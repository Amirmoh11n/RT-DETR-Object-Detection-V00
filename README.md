# RT-DETR Fine-Tuning for Object Detection

## Overview

This project focuses on fine-tuning the RT-DETR (Real-Time Detection Transformer) model for object detection using the Pascal VOC 2012 dataset.

The objective was to build a complete object detection pipeline based on a modern Vision Transformer architecture, adapt a pretrained model to a new dataset through transfer learning.

## Model

**RT-DETR (Real-Time Detection Transformer)**

RT-DETR is a transformer-based object detection architecture designed to deliver both high accuracy and efficient inference. Instead of training a detector from scratch, a pretrained RT-DETR model was fine-tuned on Pascal VOC 2012.

## Dataset

**Pascal VOC 2012**

The model was trained on the Pascal VOC 2012 object detection dataset containing 20 object categories, including:

* Person
* Car
* Bus
* Bicycle
* Train
* Dog
* Cat
* Horse
* Chair
* Bottle

and other common object classes.

## Project Goals

* Fine-tune a pretrained RT-DETR model
* Learn and apply Vision Transformer-based object detection
* Build a custom training pipeline using PyTorch
* Organize the project into modular software components
* Separate dataset handling, model loading, training, and inference logic
* Perform object detection on image data

## Project Structure

```text
RTDETR-Object-Detection/

├── dataset/
│   ├── images/
│   └── annotations/
│
├── models/
│
├── training/
│
├── inference/
│
├── outputs/
│
├── requirements.txt
│
└── README.md
```

## Training Workflow

Pretrained RT-DETR

→ Pascal VOC 2012

→ Data Processing

→ Fine-Tuning

→ Model Export

→ Image Inference

## Current Status

Completed:

* Dataset preparation
* RT-DETR integration
* Fine-tuning pipeline
* Model training
* Image inference workflow
* Local project migration from Google Colab to VS Code
* Modular project organization

Planned:

* Evaluation metrics (mAP)
* Video inference pipeline
* Real-time webcam detection
* Model optimization
* ONNX export

## Learning Outcomes

This project provided hands-on experience with:

* Vision Transformer architectures
* Transformer-based object detection
* Transfer learning and fine-tuning
* Dataset preparation and annotation handling
* Deep learning project structuring
* PyTorch-based training workflows

## Acknowledgments

* RT-DETR Authors
* Pascal VOC Dataset Contributors
* Hugging Face
* PyTorch Community
