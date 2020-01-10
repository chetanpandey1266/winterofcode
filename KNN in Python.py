#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


ds = pd.read_csv(r"E:\Winter of code\mnist_train.csv")
ds.head()


# In[3]:


x_train = ds.iloc[:, 1:].values
y_train = ds.iloc[:,0].values
#here actually we are distinguishing the dependent and the independent variables


# In[4]:


import statistics
import math
def Euclidean(x1,x2):
    return math.sqrt(np.sum((x1-x2)**2)) # x1 is input feature vector and x2 is feature vector


# In[5]:


class KNN:
    def __init__(self,x_train,x1,k = 10):
        self.k = k
        self.x_train = x_train
        self.x1 = x1
        
    def knn(self):
        """
        In this function actually I have first calculated the euclidean distance between the input and all the training 
        example. Stored this in num. To obtain the distances in ascending order I have used num1. Iterating over the first k 
        elements of num1 I found the index of these datapoints in num and used it to access the corresponding elements of
        y_train and store it in count
        
        """
        num = []
        count = []
        for i in range(y_train.size):
            num += [Euclidean(self.x_train[i],self.x1)]  # here we are creating an list which contain euclidean distance from the input point
        num1 = sorted(num) # this list contains euclidean distances in ascending form
        for j in range(self.k):  
            count += [y_train[num.index(num1[j])]] # using num we are getting the index of first k shortest distance 
        return statistics.mode(count)  # finally returning the most occuring value
            

