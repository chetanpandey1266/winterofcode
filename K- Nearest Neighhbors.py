#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


ds = pd.read_csv(r"E:\progrmming lectures\Course Material\[FreeTutorials.Us] Udemy - machinelearning\13 K-Nearest Neighbors (K-NN)\K_Nearest_Neighbors\Social_Network_Ads.csv")
ds.head()


# In[8]:


#retrived values should be converted into array
X = ds.iloc[:,2:5].values


# In[9]:


import math
def Euclidean(x1,x2,y1,y2):
    return math.sqrt((x1-x2)**2+(y1-y2)**2)


# In[10]:


import statistics


# In[11]:


def KNN(age,salary,K=5):
    num = []
    count = []
    for i in range(int(400)):
        num += [Euclidean(age,X[i][0],salary,X[i][1])]
    num1 = num;
    num1.sort(reverse = False)
    for j in range(K):
        count += [X[num.index(num1[j])][2]]
    return statistics.mode(count)    


# In[ ]:


#this does not require graph

