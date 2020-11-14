import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('default')

from sklearn import datasets
#get the iris dateset
iris = datasets.load_iris()

x = iris.data
y0 = iris.target
y = np.zeros(450).reshape(150,3)
#reshape the target,let(t->(1,0,0)or(0,1,0)or(0,0,1))
for i in range(np.shape(y0)[0]):
    if(y0[i]==0):
        y[i,0] = 1
    elif(y0[i]==1):
        y[i,1] = 1
    elif(y0[i]==2):
        y[i,2] = 1    

# binary regression
x1 = x[0:100,:]
y1 = y[0:100,:]
#training data and the testing data
x1_train = np.ones(80*5).reshape(80,5)
y1_train = np.ones(80*3).reshape(80,3)
x1_test  = np.ones(20*5).reshape(20,5)
y1_test  = np.ones(20*3).reshape(20,3)

x1_train[0:40,1:5]  = x1[0:40,:]
x1_train[40:80,1:5] = x1[50:90,:]
x1_test[0:10,1:5]   = x1[40:50,:]
x1_test[10:20,1:5]  = x1[90:100,:]
y1_train[0:40,:]    = y1[0:40,:]
y1_train[40:80,:]   = y1[50:90,:]
y1_test[0:10,:]     = y1[40:50,:]
y1_test[10:20,:]    = y1[90:100,:]
#weight and learning tate
w1 = np.ones(5*1).reshape(1,5)
theta = 0.01
#logistic regression
for i in range(100):
    theta_E = np.zeros(5*1).reshape(1,5)
    for i in range(y1_train.shape[0]):
        theta_E = theta_E + (1/(1+np.exp(-np.sum(w1*x1_train[i])))-y1_train[i,0])*x1_train[i]
    w1 = w1 - theta*theta_E
#precision
regression_precision = 0
for i in range(y1_test.shape[0]):
    p = 1/(1+np.exp(-np.sum(w1*x1_test[i])))
    if((p>0.5)&(y1_test[i,0]==1)):
        regression_precision = regression_precision + 1
    elif((p<0.5)&(y1_test[i,0]==0)):
        regression_precision = regression_precision + 1
regression_precision = regression_precision / y1_test.shape[0]
print("binary regression precision is : ",regression_precision)
    
# multiclass regression
x2 = x
y2 = y
#training data and the testing data
x2_train = np.ones(120*5).reshape(120,5)
y2_train = np.ones(120*3).reshape(120,3)
x2_test  = np.ones(30*5).reshape(30,5)
y2_test  = np.ones(30*3).reshape(30,3)

x2_train[0:40,1:5]   = x2[0:40,:]
x2_train[40:80,1:5]  = x2[50:90,:]
x2_train[80:120,1:5] = x2[100:140,:]
x2_test[0:10,1:5]    = x2[40:50,:]
x2_test[10:20,1:5]   = x2[90:100,:]
x2_test[20:30,1:5]   = x2[140:150,:]
y2_train[0:40,:]     = y2[0:40,:]
y2_train[40:80,:]    = y2[50:90,:]
y2_train[80:120,:]   = y2[100:140,:]
y2_test[0:10,:]      = y2[40:50,:]
y2_test[10:20,:]     = y2[90:100,:]
y2_test[20:30,:]     = y2[140:150,:]
#weight and learning tate
w2 = np.ones(5*3).reshape(3,5)
theta = 0.001
#logistic regression
for i in range(1000):
    theta_E = np.zeros(5*1).reshape(1,5)
    for i in range(y2_train.shape[0]):
        y_cal = np.exp(np.sum(w2[0]*x2_train[i]))/(np.exp(np.sum(w2[0]*x2_train[i]))+np.exp(np.sum(w2[1]*x2_train[i]))+np.exp(np.sum(w2[2]*x2_train[i])))
        theta_E = theta_E + (y_cal-y2_train[i,0])*x2_train[i]
    w2[0] = w2[0] - theta*theta_E
    theta_E = np.zeros(5*1).reshape(1,5)
    for i in range(y2_train.shape[0]):
        y_cal = np.exp(np.sum(w2[1]*x2_train[i]))/(np.exp(np.sum(w2[0]*x2_train[i]))+np.exp(np.sum(w2[1]*x2_train[i]))+np.exp(np.sum(w2[2]*x2_train[i])))
        theta_E = theta_E + (y_cal-y2_train[i,1])*x2_train[i]
    w2[1] = w2[1] - theta*theta_E
    theta_E = np.zeros(5*1).reshape(1,5)
    for i in range(y2_train.shape[0]):
        y_cal = np.exp(np.sum(w2[2]*x2_train[i]))/(np.exp(np.sum(w2[0]*x2_train[i]))+np.exp(np.sum(w2[1]*x2_train[i]))+np.exp(np.sum(w2[2]*x2_train[i])))
        theta_E = theta_E + (y_cal-y2_train[i,2])*x2_train[i]
    w2[2] = w2[2] - theta*theta_E
 #precision
p = np.zeros(30*3).reshape(30,3)    
for i in range(y2_test.shape[0]):
    p[i,0] = np.exp(np.sum(w2[0]*x2_test[i]))/(np.exp(np.sum(w2[0]*x2_test[i]))+np.exp(np.sum(w2[1]*x2_test[i]))+np.exp(np.sum(w2[2]*x2_test[i])))
for i in range(y2_test.shape[0]):
    p[i,1] = np.exp(np.sum(w2[1]*x2_test[i]))/(np.exp(np.sum(w2[0]*x2_test[i]))+np.exp(np.sum(w2[1]*x2_test[i]))+np.exp(np.sum(w2[2]*x2_test[i])))
for i in range(y2_test.shape[0]):
    p[i,2] = np.exp(np.sum(w2[2]*x2_test[i]))/(np.exp(np.sum(w2[0]*x2_test[i]))+np.exp(np.sum(w2[1]*x2_test[i]))+np.exp(np.sum(w2[2]*x2_test[i])))

regression_precision = 0
for i in range(p.shape[0]):
    if(((p[i,0]>p[i,1])&(p[i,0]>p[i,2]))&(y2_test[i,0]==1)):
        regression_precision = regression_precision + 1
    elif(((p[i,1]>p[i,0])&(p[i,1]>p[i,2]))&(y2_test[i,1]==1)):
        regression_precision = regression_precision + 1
    elif(((p[i,2]>p[i,0])&(p[i,2]>p[i,1]))&(y2_test[i,2]==1)):
        regression_precision = regression_precision + 1
regression_precision = regression_precision / y2_test.shape[0]
print("multi regression precision is : ",regression_precision)















    