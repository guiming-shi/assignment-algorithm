import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def logistic_regression_binary(x_train,y_train,x_test,y_test,theta,echo):
    w1 = np.ones(5*1).reshape(1,5)
    for i in range(echo):
        theta_E = np.zeros(5*1).reshape(1,5)
        for i in range(y1_train.shape[0]):
            theta_E = theta_E + (1/(1+np.exp(-np.sum(w1*x1_train[i])))-y1_train[i,0])*x1_train[i]
        w1 = w1 - theta*theta_E
