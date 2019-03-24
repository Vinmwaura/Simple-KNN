# Simple-KNN
Simple KNN implementation for Iris dataset.

K Nearest Neighbour(KNN) is a classification/supervised learning algorithm that is also a non-parametric and lazy-learning algorithm.Non-parametric means it doesn't make any assumptions on the underlying data distribution and lazy learning means it doesn't use the traing algorithm to do any generalization. 

K represents the number of training data points lying on proximity to the test data point which is used to find the class for test data.

Algorithm:
1. Load the training and test data 
2. Choose the value of K 
3. For each point in test data:
4.    For each point in training data
5.        Find the Euclidean distance between point in testing data point and training data point.
6.        Store the Euclidean distances in a list.
7.    Sort Euclidean distance list and choose the first K points closest to the testing data point.
8.    Assign a class to the test point based on the majority of classes present in the chosen K points 
