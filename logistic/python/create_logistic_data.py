import numpy as np
import random
import csv
from math import exp

data = []
DATA_NUM = 5000
TIMESTEP = 212

initial = 0.2

for i in range(0, DATA_NUM):
    #   Create one pair of time series data by using Logistic Equation
    divergence_flag = True
    while(divergence_flag):
        divergence_flag = False
        b = random.randint(0,100)
        X = []
        Y = []
        #X.append(random.random())   #   initial value
        #Y.append(random.random())   #   initial value
        X.append(initial)
        Y.append(initial)
        for t in range(1, TIMESTEP):
            X.append(3.9 * X[t-1] * (1 - X[t-1] - b / 1000.0 * Y[t-1]))
            Y.append(3.7 * Y[t-1] * (1 - Y[t-1] - 0 * X[t-1]))
            if(X[t] > 10000 or X[t] < -10000):
                divergence_flag = True
    data.append([X, Y])

npdata = np.array(data)
np.save('../npy/input_logistic_data.npy', npdata)
