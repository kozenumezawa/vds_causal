import numpy as np
import random
import csv
from math import floor



TIMESTEP = 212

initial = 0.2
b = 0
data = []
while(b < 0.2):
    name = '../csv/inputdata_b_0_' + str(int(floor(b*10))) + '.csv'
    f = open(name, 'w')
    writer = csv.writer(f)
    divergence_flag = True
    while(divergence_flag):
        print(b)
        divergence_flag = False
        X = []
        Y = []
        X.append(initial)
        Y.append(initial)
        for t in range(1, TIMESTEP):
            X.append(3.9 * X[t-1] * (1 - X[t-1] - b * Y[t-1]))
            Y.append(3.7 * Y[t-1] * (1 - Y[t-1] - 0 * X[t-1]))

            if(X[t] > 10000 or X[t] < -10000):
                divergence_flag = True
    data.append([X, Y])
    writer.writerows([X])
    writer.writerows([Y])
    f.close()
    b = (floor(b * 10) + 1) / 10
npdata = np.array(data)
np.save('../npy/input_test_data.npy', npdata)
