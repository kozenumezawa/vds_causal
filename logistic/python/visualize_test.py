# coding: utf-8

import matplotlib.pyplot as plt
import numpy
import tensorflow as tf

# load test data
testrawdata = numpy.load('../npy/test_data.npy')
testdata = numpy.zeros((testrawdata.shape[0], testrawdata.shape[2] * testrawdata.shape[1]), dtype=numpy.float32)
for i in range(testrawdata.shape[0]):
    testdata[i,:testrawdata.shape[2]] = testrawdata[i, 0]       #   X
    testdata[i, testrawdata.shape[2]:] = testrawdata[i, 1]      #   Y

# load trainning data
rawdata = numpy.load('../npy/input_logistic_data.npy')
inputdata = numpy.zeros((rawdata.shape[0], rawdata.shape[2] * rawdata.shape[1]), dtype=numpy.float32)
for i in range(rawdata.shape[0]):
    inputdata[i,:rawdata.shape[2]] = rawdata[i, 0]       #   X
    inputdata[i, rawdata.shape[2]:] = rawdata[i, 1]      #   Y

BATCH_SIZE = 1
TIME_STEP_2 = inputdata.shape[1]

W1 = numpy.load('../npy/result_W1.npy')
b1 = numpy.load('../npy/result_b1.npy')
W2 = numpy.load('../npy/result_W2.npy')
b2 = numpy.load('../npy/result_b2.npy')

times = [i for i in range(TIME_STEP_2)]
DATANUM = testdata.shape[0]

xW = numpy.zeros((DATANUM, b1.shape[0]), dtype=numpy.float32)
y_ = numpy.zeros((DATANUM, b1.shape[0]), dtype=numpy.float32)
y_activate = numpy.zeros((DATANUM, b1.shape[0]), dtype=numpy.float32)
y = numpy.zeros((DATANUM, b2.shape[0]), dtype=numpy.float32)

# visualize input and output
for i in range(0, DATANUM):
    xW[i] = numpy.matmul(testdata[i], W1)
    y_[i] = xW[i] + b1
    y_activate[i] = y_[i] / (1 + numpy.absolute(y_[i]))
    y[i] = numpy.maximum(numpy.matmul(y_activate[i], W2) + b2, 0)
plt.show()

# visualize Wx
number = [i for i in range(b1.shape[0])]
plt.plot(number, xW[4], color='r', lw=2)
plt.plot(number, xW[5], color='c', lw=2)
plt.plot(number, xW[6], color='g', lw=2)
plt.plot(number, xW[7], color='m', lw=2)
plt.plot(number, xW[8], color='b', lw=2)
plt.show()

# 4, 6, 13, 16
W = numpy.transpose(W1)
# print(W[16])
# plt.plot(times, W[16], color='g', lw=2)
# plt.plot(times, W[14], color='b', lw=2)
# plt.show()


# W1_eq = numpy.transpose(W1)
# TIME_STEP = W1_eq.shape[1] / 2
# number = [i for i in range(TIME_STEP)]
#
# for i in range(b1.shape[0]):

# sigma = 0
# t = 24
# for i in range(b1.shape[0]):
#     sigma += W1_eq[i][t-1] * W1_eq[i][TIME_STEP+t-1]
# print(sigma)


# visualize b1
# plt.plot(number, b1, color='b', lw=2)
# plt.show()
