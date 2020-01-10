#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[10]:


ds = pd.read_csv(r"E:\Winter of code\mnist_train_small.csv")
ds.head()


# In[11]:


x_train = ds.iloc[:, 1:].values
y_train = ds.iloc[:,0].values
#here actually we are distinguishing the dependent and the independent variables


# In[12]:


def updpre(x_train,y_train,y_p,w1,w0):  # this function is to update the weights 
    for i in range((y_train).size):
        y_p[i] = (x_train[i]).dot(w1) + (w0).dot(np.ones(784))
            
def difCost1(y_p,y_train,x_train,alpha):   # this method differentiate the cost function wrt w1
    m = (y_train).size
    sq = (1/m)*alpha*((np.transpose(x_train)).dot(y_p-y_train))
    return sq
    
def difCost2(y_p,y_train,x_train,alpha): # this method differentiate the cost function wrt w0
    m = (y_train).size
    sq = (1/m)*alpha*((np.transpose(x_train)).dot(np.ones((y_p-y_train).shape)))
    return sq


# In[84]:


import math
class Linear_Regression_train:
    def __init__(self,alpha = 0.001, iters = 100):
        self.alpha = alpha
        self.iters = iters
        self.x_trains = ds.iloc[:, 1:].values/3000
        self.y_trains = ds.iloc[:,0].values
        self.w0 = np.full((784),60000,dtype = "float")
        self.w1 = np.random.random(((self.x_trains[1]).size))
        self.y_p = np.zeros(((self.y_trains).size))  #predicted y
        
    def y_pred(self):
        xr,xc = (self.x_trains).shape  # xr is the number of rows and xc is the number of columns(Source: stackoverflow)
        #w0 = np.ones((1,))
        #w1 = np.random.random((xc))
        for i in range((self.y_trains).size):
            self.y_p[i] = (self.w1).dot(self.x_trains[i]) + (self.w0).dot(np.ones(784))  # last term written to remove ambiguity
        return self.y_p
    
    def updpre(self,x_train,y_train,y_p,w1,w0):  # this function is to update the weights 
        for i in range((self.y_train).size):
            self.y_p[i] = (self.x_train[i]).dot(self.w1) + (self.w0).dot(np.ones(784))
        return self.y_p
    
    def grad(self):  # gradient descent 
        self.y_p =self.y_pred()
        for i in range(self.iters):
            self.w1 += difCost1(self.y_pred(),self.y_trains,self.x_trains,self.alpha)
            self.w0 += difCost2(self.y_pred(),self.y_trains,self.x_trains,self.alpha)
            self.y_p = updpre(self.x_trains,self.y_trains,self.y_p,self.w1,self.w0)       
        return np.round(self.y_p)
    
    def accuracy(self):
        return np.sum(np.absolute(self.y_trains-self.y_p))/20000


# In[ ]:




