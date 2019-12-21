#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[19]:


# read data
ds = pd.read_csv(r"E:\progrmming lectures\Course Material\[FreeTutorials.Us] Udemy - machinelearning\12 Logistic Regression\LR\Social_Network_Ads.csv")
ds.head()


# In[20]:


#Distributing the dataset into dependent and independent variables
X = ds.iloc[:,1:-1].values  #as User Id is a useless data for the mathematical model hence removing it 
Y = ds.iloc[:, -1].values


# In[21]:


# Y is a 1d array so we are converting it to a 2d array
Y2 =  np.zeros((400,1))
for i in range(400):
    Y2[i][0] = Y[i]


# In[22]:


#Encoding of labels
for i in range(400):
    if X[i][0]== "Male":
        X[i][0] =1
    elif X[i][0]== "Female":
        X[i][0]= 0


# In[23]:


# Dividing the dataset into training set and test set.
n = float(input("Enter the fraction which is given to test set:"))
X_test = X[:int(400*(1-n)),:]
X_train = X[int(-400*n-1):-1,:]
Y_test = Y2[:int(400*(1-n)),:]
Y_train = Y2[int(-400*n-1):-1,:]
print(Y_test.shape,Y_train.shape)


# In[24]:


# a problem 
# solution : reference to stackoverflow
#screenshot in phone

import math

def sigmoid(x):
    if x<0:
        return 1 - 1/(1+math.exp(x))
    else:
        return 1/(1+math.exp(-x))

    


# In[25]:


#Initialising the hyper parameters
w = np.array([.02,.002,-.00000001])
w0 = 1
alpha = 0.0001


# In[26]:


#writing the wTx
h0 = np.dot(X_train,w)


# In[27]:


#writing y_predicted
y_p = np.zeros((len(h0)))
for i in range(len(h0)):
    y_p[i] = sigmoid(h0[i])


# In[28]:


#useless
#writing the cost function
def Cost():
    sum = 0
    for i in range(Y_train.size):
        sum +=-Y_train[i][0]*math.log(y_p[i],2) - (1-Y_train[i][0])*math.log(1-y_p[i],2)
    return sum

print(Cost())        


# In[29]:


# converting Y_train matrix into a 1d matrix
Y_train1 = np.zeros((Y_train.size))
for i in range(Y_train.size):
    Y_train1[i] = Y_train[i][0]


# In[30]:


def updh(w):
    h0 = np.dot(X_train,w)
    for i in range(len(h0)):
        y_p[i] = sigmoid(h0[i])


# In[31]:


print((y_p-Y_train1).shape)
print(X_train[:,2].shape)


# In[32]:


def difCost(y_p,Y_train,theta):  #differentiated cost
    m = len(Y_train1)
    return (1/m)*alpha*(np.dot(np.transpose(y_p-Y_train1),theta))


# In[40]:


def grad(iters):
    for j in range(len(y_p)):
        w[0] += difCost(y_p,Y_train1,X_train[:,0])
        updh(w)
    for j in range(len(y_p)):
        w[1] += difCost(y_p,Y_train1,X_train[:,1])
        updh(w)
    for j in range(len(y_p)):
        w[2] += difCost(y_p,Y_train1,X_train[:,2])
        updh(w)
    return w


# In[55]:


grad(1000)


# In[56]:


plt.plot(X_train*1000,Y_train,'o')
plt.plot(X_train*1000,y_p)
plt.show()


# In[ ]:




