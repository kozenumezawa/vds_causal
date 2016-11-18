import numpy as np
import random
import csv
from math import floor



TIMESTEP = 212

initial = 0.2
b = 0
data = []
while(b <= 10):
    name = '../csv/testdata_b_0_' + str(int(floor(b*10))) + '.csv'
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
            X.append(3.9 * X[t-1] * (1 - X[t-1] - b / 100.0 * Y[t-1]))
            Y.append(3.7 * Y[t-1] * (1 - Y[t-1] - 0 * X[t-1]))

            if(X[t] > 10000 or X[t] < -10000):
                divergence_flag = True
    data.append([X, Y])
    writer.writerows([X])
    writer.writerows([Y])
    f.close()
    b = b + 1
npdata = np.array(data)
np.save('../npy/test_data.npy', npdata)
