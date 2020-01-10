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


# Unsupervised 
x_train = ds.iloc[:, 1:].values


# In[4]:


import math
def Euclidean(x1,x2):
    return math.sqrt(np.sum((x1-x2)**2))


# In[12]:


import random
class Kmean:
    def __init__(self,x_train,k = 10,iters = 100):
        self.k = k
        self.x_train = x_train
        self.iters = iters
        self.num2 = []
        
    def impl(self):
        num = []
        for i in range(self.k):  # creating an array of indices
            num += [random.randint(0,19999)]
        centers =[]
        for i in range(self.k): #random initialised the clusters centers
            centers = self.x_train[num[i]]
        temp = np.ones((self.k,19999))  # temp will store in each column for 60000 elements
        for i in range(self.iters):  # Main Algo Starts
            for i in range(self.k):
                for j in range(19999):
                    temp[i,j] = Euclidean(self.x_train[j],centers[i])  # temp ki har row mein pehle centroid se har training example
                    #ke distance hai......Har column mein per training example distance centroids se di hai
            num3 = []
            for i in range(19999):
                num3 += [(np.where(temp == min(temp[:,i])))[1]]  # num will be the list of the cluster centers that is closer to each training example
            num1 = []
            self.num2 = []
            for i in range(1,11):
                for j in range(19999):
                    if num3[j].any == i:
                        num1 += [j]
                self.num2 += [num1]  # list of list of training example which is close to each 
                num1 = []
            centers1 = []
            for i in range(10):
                centers1 += [np.mean((self.num2[i]))]
            if centers1 == centers:
                return centers1
            centers = centers1
        return centers  


# In[ ]:


K = Kmean(x_train)
K.impl()


# In[ ]:





# In[ ]:




