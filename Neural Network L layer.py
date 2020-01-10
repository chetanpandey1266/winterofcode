#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


ds = pd.read_csv(r"E:\Winter of code\mnist_train_small.csv")
ds.head()


# In[3]:


label = ds.iloc[:,0].values
z = ds.iloc[:,1:].values
# input layer created
z[:,1].size


# In[4]:


import math
def sigmoid(x):
    if x.all():   # all returns true if all elements are non zero or else returns zero
        return 1/(1+np.exp(-x))
    else:
        return 1- 1/(1+np.exp(x)) 


# In[5]:


def derivative(output):
        return output*(1-output)


# In[6]:


def y_original():
    yo = np.zeros((z[:,1].size,10))
    # Creating an array which tells that at which position in each row zero should be replaced by one
    n = label - 1
    for i in range(z[:,1].size):  # while creating n we have made 0 to -1 and so to rectify this we are doing this
        if n[i]<0:
            n[i] = 1
    for i in range(z[:,1].size):  # finallly creating y_predicted
        yo[i][label[i]] = 1
    return yo


# In[12]:


import random
class l_layer:
    def __init__(self,label,z1,n = 5,iters = 1000):
        self.label = label
        self.z1 = z1
        self.iters = iters
        self.n = n
        self.num = []
        self.w_list = []
        self.z_list = []
        self.a_list = []
        self.delta = []
        self.der = []
        
    def forward(self):
        t = 784
        #DEFINING THE NUMBER OF NEURONS IN EACH LAYER
        # defining the number of neurons in layers except  input
        for i in range(self.n-1):  # Creating a list having number of neurons for each layer
            temp = random.randint(100,t)
            self.num += [temp]
            t = temp   # this is done to decrease the number of neurons in each layer as we move towards the output
            if self.num[i] == 100:
                self.num[i] = [random.randint(10,100)]
        self.num += [10] # this is to ensure that output layer contains 10 neurons
        #CREATING LIST OF WEIGHTS
        for i in range(self.n-1):  # Creating a list weights for each layer 
            if i > 0:
                self.w_list += [np.random.random((self.num[i-1],self.num[i]))]# This is done to ensure that previous weights matrix columns is equal to next weights matrix row
            else:
                self.w_list += [np.random.random((784,self.num[i]))]
        self.w_list += [np.random.random((self.num[len(self.num)-2],10))]  # Finally creating weights for output layer
        # CREATING LIST OF Z
        self.z_list += [self.z1]
        for i in range(len(self.w_list)): # List of Z. In z_list we have also included the input. So its len is 6
            self.z_list += [np.dot(self.z_list[i],self.w_list[i])]
        for i in range(len(self.z_list)):
            self.z_list[i] = self.z_list[i]/1000
        # CREATING A LIST OF ACTIVATION FUNCTION
        for i in range(1,len(self.z_list)): # 1 because at zero index z_list is having z1
            self.a_list += [sigmoid(self.z_list[i])] # Each element of z_list is vector containig z for each layer
        # now a_list is a list of vector containing a value for each neuron in each layer
    
    def update(self,w_list):
        self.z_list = []
        self.a_list = []
        self.z_list += [self.z1]
        for i in range(len(self.w_list)): # List of Z. In z_list we have also included the input. So its len is 6
            self.z_list += [np.dot(self.z_list[i],self.w_list[i])]
        for i in range(len(self.z_list)):
            self.z_list[i] = self.z_list[i]/1000
        # CREATING A LIST OF ACTIVATION FUNCTION
        for i in range(1,len(self.z_list)): # 1 because at zero index z_list is having z1
            self.a_list += [sigmoid(self.z_list[i]-20)] # Each element of z_list is vector containig z for each layer
        # now a_list is a list of vector containing a value for each neuron in each layer
        return self.a_list
        
    def backward(self):
        for i in range(self.iters):
            # creating a list of deltas
            self.delta += [y_original()-self.a_list[len(self.a_list)-1]]
            for i in range(len(self.w_list)-1): # delta is an inverted list that is from end to first
                self.delta += [(self.delta[i].dot(self.w_list[len(self.w_list)-1-i].T))*derivative(self.a_list[len(self.w_list)-2-i])]
            # now the real backpro starts
            # A list of derivatives
            for i in range(len(self.w_list)-1):
                self.der += [(1/z[:,1].size)*(self.a_list[-1-i].T).dot(self.delta[i+1])]
            for i in range(len(self.w_list)-1):
                self.w_list[-1-i] += self.der[i].T
            self.a_list = self.update(self.w_list)
        return self.a_list
    
    def accuracy(self):
        return np.sum((np.absolute(y_original()-np.round(self.a_list[-1]))))/200


# In[14]:


L1 = l_layer(label,z,7,100)
L1.forward() 
L1.backward()
L1.accuracy()


# In[ ]:





# In[ ]:




