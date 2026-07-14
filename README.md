# RT-DETR Object Detection Fine-Tuning

## Overview

This project focuses on fine-tuning the RT-DETR (Real-Time Detection Transformer) vision transformer model for object detection tasks using the Pascal VOC 2012 dataset.

The goal of this project is to adapt a state-of-the-art transformer-based object detector to a new dataset through transfer learning and fine-tuning, while building a complete training, evaluation, and inference pipeline using PyTorch.

## Model

**RT-DETR (Real-Time Detection Transformer)**

RT-DETR is a transformer-based object detection architecture that combines the accuracy of DETR-style models with real-time inference capabilities. The model was initialized with pretrained weights and further fine-tuned on Pascal VOC 2012 to improve performance on the target dataset.

## Dataset

**Pascal VOC 2012**

The model was trained and evaluated using the Pascal VOC 2012 object detection dataset, which contains annotated images across 20 object categories, including:

* Person
* Car
* Bus
* Bicycle
* Dog
* Cat
* Train
* Horse
* And other common object classes

## Project Objectives

* Fine-tune a pretrained RT-DETR model on Pascal VOC 2012
* Build a custom PyTorch data pipeline
* Implement training and validation workflows
* Save and manage model checkpoints
* Evaluate object detection performance
* Perform inference on unseen images and videos
* Prepare the model for real-world deployment scenarios

## Technologies Used

* Python
* PyTorch
* Transformers (Hugging Face)
* OpenCV
* NumPy
* Matplotlib
* Google Colab

## Training Pipeline

Dataset Preparation

→ Annotation Processing

→ RT-DETR Data Formatting

→ Fine-Tuning

→ Validation

→ Checkpoint Saving

→ Model Export

→ Inference

## Results

The project demonstrates the successful adaptation of a pretrained RT-DETR model to a new object detection dataset through fine-tuning.

The resulting model is capable of detecting multiple object categories and generating bounding box predictions on previously unseen images.

## Future Improvements

* Full mAP evaluation
* Hyperparameter optimization
* Larger-scale training
* Real-time webcam inference
* Traffic monitoring applications
* Deployment using ONNX or TensorRT

## Repository Structure

```text
RTDETR-Object-Detection/

├── notebooks/
├── checkpoints/
├── models/
├── inference/
├── outputs/
├── dataset/
├── README.md
└── requirements.txt
```

## Acknowledgments

* RT-DETR Authors
* Hugging Face Transformers
* Pascal VOC Dataset Contributors
* PyTorch Community
