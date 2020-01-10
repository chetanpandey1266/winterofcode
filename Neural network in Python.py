#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[13]:


ds = pd.read_csv(r"E:\Winter of code\mnist_train_small.csv")
ds.head()


# In[20]:


label = ds.iloc[:,0].values
z = ds.iloc[:,1:].values
# input layer created


# In[21]:


def sigmoid(x):
    if x.all():
        return 1/(1+math.exp(-x))
    else:
        return 1- 1/(1+math.exp(x)) 


# In[22]:


def derivative(output):
        return output*(1-output)


# In[23]:


def y_original():
    yp = np.zeros((z[:,1].size,10))
    # Creating an array which tells that at which position in each row zero should be replaced by one
    n = label - 1
    for i in range(z[:,1].size):  # while creating n we have made 0 to -1 and so to rectify this we are doing this
        if n[i]<0:
            n[i] = 1
    for i in range(z[:,1].size):  # finallly creating y_train in vectorized form
        yp[i][label[i]] = 1
    return yp    


# In[ ]:


import math
class neural_network_train:
    def __init__(self,label,z1,iters =10):  # this will be used for setting data in the variables
        self.iters = iters
        self.label = label
        self.z1 = z1
        self.z1 = self.z1/300
        self.w1 = np.random.random((784,400))  # 400 becoz the hidden layer consist of only 400 activation units
        #Creating z2
        self.z2 = (self.z1).dot(self.w1)
        self.a1 = np.ones((z[:,1].size,400))
        self.w2 = np.random.random((400,10))
        # Creating z3
        self.z3 = (self.z2).dot(self.w2)
        self.a2 = np.ones((z[:,1].size,10))
        self.loss = np.zeros((z[:,1].size,10))
        self.yp = y_original()
        
    
    def forward(self):   # creating activation function for first layer
        #Feature Scaling
        self.z2 = self.z2/20000
        for i in range(z[:,1].size):
            for j in range(400):
                self.a1[i,j] = sigmoid(self.z2[i,j]-44)
        # feature scaling on z3
        self.z3 = self.z3/200
        for i in range(z[:,1].size):
            for j in range(10):
                self.a2[i][j] = sigmoid(self.z3[i][j]-44)
                
    def update(self):
        self.z2 = (self.z1).dot(self.w1)
        self.z3 = (self.z2).dot(self.w2)
        for i in range(z[:,1].size):
            for j in range(400):
                self.a1[i,j] = sigmoid(self.z2[i,j]-44)
        for i in range(z[:,1].size):
            for j in range(10):
                self.a2[i][j] = sigmoid(self.z3[i][j]-44)
        return self.a2

    
    def backpro(self):
        for i in range(self.iters):
            dw2 = np.dot(self.a1.T,(2*(self.a2-self.yp)*derivative(self.a2)))
            dw1 = np.dot(self.z1.T,np.dot(2*(self.a2-self.yp)*derivative(self.a2),self.w2.T)*derivative(self.a1))
            self.w1 += dw1
            self.w2 += dw2
            self.a2 = self.update()
        return self.a2  
    
    
    def accuracy(self):
        return np.sum(np.absolute(self.yp - self.a2))/z[:,1].size


# In[ ]:


NN = neural_network_train(label,z,1000)
NN.forward()
NN.backpro()
NN.accuracy()


# In[ ]:





# In[ ]:




