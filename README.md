# Introductory ML Visualizations 
This repo contains modules for educational visualizations of ML techniques.
Linear regression with gradient descent, softmax &
logistic regression for double and multiclass classification, and full batch
gradient descent, SGD, z-scoring, and other techniques for optimization are
hand implemented and displayed through use of NumPy and Matplotlib.
Modules contain classes and functions with varying levels of 
reusability, documented within module docstrings and beneath. 

###### softmax_regression
Contains SoftmaxRegression class for training and classification of
general n-dimensional data. Performs z-scoring normalization and full batch 
gradient descent across all passed training examples. Online, single
example classification is available following training. Running module 
as main generates, plots, trains, and classifies 3d data. See module for
documentation and example of class usage. Data may be plotted with various
graphical options and print outs, as shown beneath. 

###### logistic_regression
Contains functions for training and classification through logistic regression
using full SGD. Running module as main generates 3d visualizations of logistic 
classifier, including decision boundary graphing. See main for further options
and examples of function use. 

###### random_data
Contains two generators for creation of uniformly distributed 3d and 2d data,
generated from underlying planar or linear functions. Noise, placement, and
distribution of data can be modulated via parameters. Further documentation in
module.

###### linear_regression
Running module displays simple visualization of gradient descent for linear
regression, fitting a line to noisy 2d data generated from an underlying linear
distribution.
Implementations of Softmax, Logistic, and Linear Regression with gradient
descent





