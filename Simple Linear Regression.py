#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


# In[3]:


ds = pd.read_csv(r"E:\progrmming lectures\Course Material\[FreeTutorials.Us] Udemy - machinelearning\04 Simple Linear Regression\Simple_Linear_Regression\Salary_Data.csv")
ds.head()


# In[4]:


X = ds.iloc[:, :-1].values
Y = ds.iloc[:,1].values
#here actually we are distinguishing the dependent and the independent variables


# In[5]:


num = len(X) 
n = float(input())
X_train = X[:int(num*(1-n))]
X_test = X[int(-num*n):num]
Y_train = Y[:int(num*(1-n))]
Y_test = Y[int(-num*n):num]


# In[6]:


#setting hyper parameters
alpha = 0.0001
#iters = 1000  #number of iterations
theta = np.array([25000,1.0])  # we are taking theta as a row matrix because  it has two values, that is, theta0 and theta1


# In[7]:



y_p = np.zeros((Y_train.size))
for i in range(len(X_train)):
    y_p[i] = X_train[i] +1.0
def updpre():
    for i in range(len(X_train)):
        y_p[i] = X_train[i]*theta[1]+theta[0]
    return y_p


# In[8]:


m = len(Y_train)
def difCost1(y_p,Y_train):
    sq = (1/m)*alpha*((y_p-Y_train).dot(X_train))
    return sq
def difCost2(y_p,Y_train):
    sq = (1/m)*alpha*((y_p-Y_train).dot(np.ones(X_train.shape)))
    return sq


# In[9]:


from math import *
def diff(func, value):
    h = 0.0000000001
    top = function(value+h)-function(value)
    bottom = h
    slope = top/bottom
    return round(slope,3)

# seems to be useless


# In[12]:


def Grad(iters):
    for i in range(iters):
        theta[1] += difCost1(Y_train,y_p)
        theta[0] += difCost2(Y_train,y_p)
        updpre()       
    return theta[1],theta[0]


Grad(100000)
print(theta[1],theta[0])    
    


# In[13]:



plt.plot(X_train,Y_train,'o')
plt.plot(X_train,y_p)
plt.show()


# In[ ]:





# In[ ]:




