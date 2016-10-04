import numpy as np
import random
import csv
from math import floor


TIMESTEP = 25

initial = 0.1
b = 0
data = []
while(b < 0.9):
    name = '../csv/inputdata_b_0_' + str(int(floor(b*10))) + '.csv'
    f = open(name, 'w')
    writer = csv.writer(f)
    divergence_flag = True
    while(divergence_flag):
        divergence_flag = False
        X = []
        Y = []
        X.append(initial)
        Y.append(initial)
        for t in range(1, TIMESTEP):
            X.append(X[t-1] * (3.8 - 3.8 * X[t-1] - 2 * b * Y[t-1]))
            Y.append(Y[t-1] * (3.5 - 3.5 * Y[t-1] - 0 * X[t-1]))
            if(X[t] > 10000 or X[t] < -10000):
                divergence_flag = True
        data.append([X])
        data.append([Y])
        writer.writerows([X+Y])
        f.close()
        b = (floor(b * 10) + 1) / 10
npdata = np.array(data)
np.save('../npy/input_test_data.npy', npdata)
