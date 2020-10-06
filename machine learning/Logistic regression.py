import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings('default')

from sklearn import datasets

iris = datasets.load_iris()

x = iris.data
y0 = iris.target
y = np.zeros(450).reshape(150,3)

for i in range(np.shape(y0)[0]):
    if(y0[i]==0):
        y[i,0] = 1
    elif(y0[i]==1):
        y[i,1] = 1
    elif(y0[i]==2):
        y[i,2] = 1    


x1 = x[0:100,:]
y1 = y[0:100,:]

x1_train = np.ones(80*5).reshape(80,5)
y1_train = np.ones(80*3).reshape(80,3)
x1_test  = np.ones(20*5).reshape(20,5)
y1_test  = np.ones(20*3).reshape(20,3)

x1_train[0:40,1:5]  = x1[0:40,:]
x1_train[40:80,1:5] = x1[50:90,:]
x1_test[0:10,1:5]   = x1[40:50,:]
x1_test[10:20,1:5]  = x1[90:100,:]
y1_train[0:40,:]  = y1[0:40,:]
y1_train[40:80,:] = y1[50:90,:]
y1_test[0:10,:]   = y1[40:50,:]
y1_test[10:20,:]  = y1[90:100,:]

w1 = np.ones(5*1).reshape(1,5)
theta = 0.0001
for i in range(1000):
    theta_E = np.zeros(5*1).reshape(1,5)
    for i in range(y1_train.shape[0]):
        theta_E = theta_E + (w1*x1_train[i]-y1_train[i,0])*x1_train[i]
    w1 = w1 - theta*theta_E


for i in range(y1_train.shape[0]):
    p = 1/(1+np.exp(-np.sum(w1*x1_train[i])))
    print(p)