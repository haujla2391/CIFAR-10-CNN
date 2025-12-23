# CIFAR-10 Image Classification with CNNs in PyTorch

This project demonstrates how to build, train, and evaluate a **Convolutional Neural Network (CNN)** for classifying images from the **CIFAR-10** dataset using **PyTorch**.

## About

CIFAR-10 is a widely used dataset for image classification, consisting of **60,000 32x32 color images** in **10 classes**, with 6,000 images per class.  
This project implements a CNN to learn meaningful features from these images and predict their class.

## Dataset

The dataset can be automatically downloaded using `torchvision.datasets.CIFAR10`. It includes:

- **10 classes**: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`  
- **50,000 training images** and **10,000 test images**

## Model Architecture

The CNN model consists of:

- **Convolutional layers** with ReLU activation  
- **Max pooling layers** for downsampling  
- **Fully connected layers** for classification  
- **Dropout layers** to reduce overfitting  
