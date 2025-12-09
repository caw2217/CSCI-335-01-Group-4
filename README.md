# CIFAR-10 Image Classification

by Cameron Wilson (caw2217) and Brandon Santore (bms1276)

## Abstract
The goal of this project is to evaluate three different models' performance on classifying the CIFAR-10 Dataset.

## Neural Network
NeuralNetwork.py is the neural network model. It contains the main CNN model used in this project. Our CNN uses VGG-like architecture with two convolution pass blocks with max pooling in between. Two of these blocks are used.

## Traditional Models
TraditionalModels.py trains and evaluates two traditional models, a Random Forest and a SVM. For both models, the CIFAR-10 dataset is preprocessed and features are extracted using a Histogram of Oriented Gradients (HOG) method. This is fed into each model for training. 
