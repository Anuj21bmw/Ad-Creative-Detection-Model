# Ad-Creative-Detection-Model
Ad Creative Recognition with Computer Vision 

This repository contains code for a computer vision model that recognizes ad creatives using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The model is trained to classify images into two categories: 'ad creative' and 'non-ad creative'.
Introduction
Ad creative recognition is crucial for digital marketing to automate the process of identifying and categorizing ad creatives. This project uses a deep learning approach to classify images into 'ad creative' and 'non-ad creative' categories.

Dataset
The model is trained on the CIFAR-10 dataset, which is a collection of 60,000 32x32 color images in 10 different classes. For this project, only two classes are used: airplanes (class 0) and automobiles (class 1). These classes are relabeled to represent 'ad creative' and 'non-ad creative', respectively.

Model Architecture
The model is a Convolutional Neural Network (CNN) with the following architecture:

Conv2D layer with 32 filters and ReLU activation
MaxPooling2D layer
Conv2D layer with 64 filters and ReLU activation
MaxPooling2D layer
Conv2D layer with 128 filters and ReLU activation
MaxPooling2D layer
Flatten layer
Dense layer with 512 units and ReLU activation
Dropout layer with a rate of 0.5
Dense output layer with sigmoid activation
Training
The model is trained with the following configuration:

Loss function: Binary Crossentropy
Optimizer: Adam
Metrics: Accuracy
Early stopping is applied with a patience of 3 epochs on the validation loss.
Evaluation
The model is evaluated on a test set to determine its accuracy. Additionally, the training history is plotted to visualize the accuracy and loss over epochs.

Prediction
A function is provided to predict if a given image is an ad creative. The image is preprocessed to match the input shape of the model, normalized, and then passed through the model to get the prediction.

Requirements
The following packages are required to run the code:

TensorFlow
OpenCV
Matplotlib
NumPy

Results
The model achieves an accuracy of approximately 94% on the test set, demonstrating its ability to differentiate between ad creatives and non-ad creatives.
