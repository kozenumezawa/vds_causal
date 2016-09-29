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

y_ = numpy.matmul(testdata[1], W1) + b1
y_activate = y_ / (1 + numpy.absolute(y_))
y = numpy.maximum(numpy.matmul(y_activate, W2) + b2, 0)

print(y.shape)
plt.subplot(2, 1, 1)
plt.plot(times, testdata[5], color='r', lw=2)
plt.plot(times, y, color='g', lw=1)
#
# plt.subplot(2, 1, 2)
# output = y.eval(session=sess, feed_dict={x: [testdata[8]], keep_prob: 1.0})
# plt.plot(times, testdata[8], color='r', lw=2)
# plt.plot(times, output[0], color='g', lw=1)

# plt.subplot(3, 2, 3)
# plt.subplot(3, 2, 4)
# plt.subplot(3, 2, 5)
# plt.subplot(3, 2, 6)
plt.show()
