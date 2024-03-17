# CS6910_assignment1

Goal of this assignment is to implement a feed-forward neural network from scratch using numpy or pandas

## Problem Statement
In this assignment you need to implement a feedforward neural network and write the backpropagation code for 
training the network. We strongly recommend using numpy for all matrix/vector operations. You are not allowed 
to use any automatic differentiation packages. This network will be trained and tested using the 
Fashion-MNIST dataset. Specifically, given an input image (28 x 28 = 784 pixels) from the Fashion-MNIST dataset, 
the network will be trained to classify the image into 1 of 10 classes.

## Process
* X_train, y_train, X_test, y_test was loaded using the fashion mnist dataset
* 10 % data of train is given as validation dataset
* The weight initialisation is given using the if and elif condition. If the given weight intialisation is not there, it will give out the 'value error'
* 


