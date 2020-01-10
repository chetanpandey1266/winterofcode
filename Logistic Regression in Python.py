#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


ds = pd.read_csv(r"E:\Winter of code\mnist_train_small.csv")
ds.head()


# In[ ]:


x_train = ds.iloc[:, 1:].values
y_train = ds.iloc[:, 0].values
#here actually we are distinguishing the dependent and the independent variables


# In[4]:


def sigmoid(x):    # Source: Stackoverflow
    if all(x):    # all returns true if all elements are non zero or else returns zero
        return 1 - 1/(1+np.exp(x))
    else:
        return 1/(1+np.exp(-x))


# In[5]:


def y_original():
    yo = np.zeros((y_train.size,10))
    # Creating an array which tells that at which position in each row zero should be replaced by one
    n = y_train - 1
    for i in range(y_train.size):  # while creating n we have made 0 to -1 and so to rectify this we are doing this
        if n[i]<0:
            n[i] = 1
    for i in range(y_train.size):  # finallly creating y_train in vectorized form
        yo[i][y_train[i]] = 1
    return yo # y_original    


# In[6]:


def difCost(w1,dcost,alpha,y_p):
    m = len(y_train)
    for i in range(y_train.size):
        dcost = ((1/m)*alpha*((np.sum(y_p[1]-y_original()[1]))*(w1)))
    return dcost


# In[7]:


import math
class Logistic_Regression_train:
    def __init__(self,alpha = 0.001, iters = 500):
        self.alpha = alpha
        self.iters = iters
        self.x_train = ds.iloc[:, 1:].values
        self.y_train = ds.iloc[:, 0].values
        self.y_p = np.zeros( ((self.y_train).size,10) )
        self.w1 = np.random.random(((self.x_train[1]).size,10)) 
        self.h0 = np.dot(self.x_train,self.w1)
        self.dcost = np.zeros(((self.w1).shape))
        
    def updh(self,y_p):     # this will take the value of y_p to be updated 
        for i in range(self.h0[:,1].size):
            self.y_p[i] = sigmoid(self.h0[i]-13156.619790000132)     
        return self.y_p
    

    # initially dcost's rows will be same but later we will update them  
    
    def Grad(self):
        for i in range(self.iters):  # iterates the given number of times and returns w1 for which cost is minimum
            for i in range(784):
                self.w1 += self.dcost[i]
            self.y_p = self.updh(self.y_p)
            self.dcost = difCost(self.w1,self.dcost,self.alpha,self.y_p)
        return self.w1
    
    def Predicted(self):
        return self.y_p 

