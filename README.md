# winterofcode

Description of each file 

## KNN in Python
class name: KNN
This class accepts three arguments from the user.
1.Training Set: The training set must be in format of matrix which contains number of training example along column and the pixels along the row.

2.Input: The input must be in form of an 1d array which contains pixel for which we need the digit that it represents.

3.Number of clusters: Also well known as k. This is a default parameter whose default value is 10. The user can change as per his\her convenience.

This class contains a method (method_name : knn ). This method do not accept any arguments. It just outputs the value predicted by the algorithm. 


## Linear Regression In Python
class_name: Linear_Regression_train:
This class accepts two arguments:
1. alpha:  Learning rate. Its default value is set to 0.001.
2. iters :  Number of iterations. Its default value is set to 100.

This class contains two methods:

1. grad(): Returns the predicted value after applying gradient descent.

2. accuracy(): Returns the accuracy of the model. 


## Logistic Regression In Python
class_name: Logistic_Regression_train:
This class accepts two arguments:
1. alpha:  Learning rate. Its default value is set to 0.001.
2. iters :  Number of iterations. Its default value is set to 500.

This module contains two methods:
1. Grad(): Returns the weight  

2.Predicted(): Returns the predicted value of y


## Neural Network in Python(two layer model)
class_name: neural_network_train(label,Z,iters)
This class accepts four arguments from the user:
1.label: This accepts the vectorized form of the desired output. 

2.z1: This accepts the training examples in which each row corresponds to the pixels for one training example.

3. iters: This accepts the number of iterations in order to train the network.

This class contains three methods:
1. forward(): This methoddo not accept any argument
2. backward(): This method returns the predicted output in vectorized format
3. accuracy(): This method returns the accuracy of model in percentage.


## Neural Network L layer( l Layer model)
class_name: l_layer(label,n,iters)
This class accepts four arguments from the user.
1.label: This accepts the vectorized form of the desired output. 

2.z1: This accepts the training examples in which each row corresponds to the pixels for one training example.
 
3.. n: This accepts the number of layers you want to have in your network. Constraint for n: n<10.

4. iters: This accepts the number of iterations in order to train the network.

This class contains three methods:
1. forward(): This method do not accept any argument
2. backward(): This method returns the predicted output in vectorized format
3. accuracy(): This method returns the accuracy of model in percentage.
