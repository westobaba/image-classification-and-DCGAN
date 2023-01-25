# image-classification-and-DCGAN
Image Classification using Logistic Regression, Tiny VGG, and DC GAN
This repository contains code for image classification using logistic regression, a baseline replica of Tiny VGG, and a Deep Convolutional Generative Adversarial Network (DC GAN) written in Python and PyTorch. The following models are included:

Logistic Regression on MNIST: A simple logistic regression model trained on the MNIST dataset for handwritten digit classification.

Tiny VGG on Brain Tumor Classification: A replica of the Tiny VGG architecture trained on brain tumor images for classification.

Pretrained Model on Cards Image: A pretrained model on cards images, you can use this as a feature extractor and fine-tune it to your use case.

DC GAN: A DC GAN model trained on images to generate new images from other images.

All the models are implemented using PyTorch, and the code is well-commented for easy understanding. You can use this repository as a starting point for your own image classification and generation projects.

Requirements
Python 3.x
PyTorch
torchvision
numpy
matplotlib
tqdm
Datasets
MNIST dataset is already available in PyTorch, you don't need to download it.
For brain tumor classification, you need to download the dataset from https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection
For card images, you need to download the dataset from https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification
