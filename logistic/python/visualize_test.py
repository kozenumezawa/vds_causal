# coding: utf-8

import matplotlib.pyplot as plt
import numpy
import tensorflow as tf

# load test data
testrawdata = numpy.load('../npy/input_test_data.npy')
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

    plt.subplot(9, 1, i+1)
    plt.plot(times, testdata[i], color='r', lw=2)
    plt.plot(times, y[i], color='g', lw=1)
plt.show()

# visualize Wx
number = [i for i in range(b1.shape[0])]
plt.plot(number, xW[0], color='r', lw=2)
plt.plot(number, xW[2], color='c', lw=2)
plt.plot(number, xW[4], color='g', lw=2)
plt.plot(number, xW[6], color='m', lw=2)
plt.plot(number, xW[8], color='b', lw=2)
plt.show()

# 3, 8, 14, 18
W = numpy.transpose(W1)
plt.plot(times, W[8], color='g', lw=2)
plt.plot(times, W[14], color='b', lw=2)
print(W[3])
print(W[8])
print(W[14])
plt.show()

# visualize b1
# plt.plot(number, b1, color='b', lw=2)
# plt.show()
