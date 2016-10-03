# coding: utf-8

import matplotlib.pyplot as plt
import numpy
import tensorflow as tf

rawdata = numpy.load('ocean.normalized.npy') # rawdata.shape = (10100, 2, 212)
data = numpy.zeros((rawdata.shape[0] * 2, rawdata.shape[2] * 2), dtype=numpy.float32)
st = numpy.zeros((rawdata.shape[0], rawdata.shape[2] * 2), dtype=numpy.float32)

for i in range(rawdata.shape[0]):
    st[i, :rawdata.shape[2]] = rawdata[i, 0]          #   S(salinity)
    st[i, rawdata.shape[2]:] = rawdata[i, 1]          #   T(water temperature)

# print(st.shape) = (10100, 424)
PIXELS = st.shape[1]  # = 424

W1 = numpy.load('../npy/result_W1.npy')
b1 = numpy.load('../npy/result_b1.npy')
W2 = numpy.load('../npy/result_W2.npy')
b2 = numpy.load('../npy/result_b2.npy')

times = [i for i in range(st.shape[1])]
DATANUM = st.shape[0]

xW = numpy.zeros((DATANUM, b1.shape[0]), dtype=numpy.float32)
y_ = numpy.zeros((DATANUM, b1.shape[0]), dtype=numpy.float32)
y_activate = numpy.zeros((DATANUM, b1.shape[0]), dtype=numpy.float32)
y = numpy.zeros((DATANUM, b2.shape[0]), dtype=numpy.float32)

# visualize input and output
for i in range(0, DATANUM):
    xW[i] = numpy.matmul(st[i], W1)
    y_[i] = xW[i] + b1
    y_activate[i] = y_[i] / (1 + numpy.absolute(y_[i]))
    y[i] = numpy.maximum(numpy.matmul(y_activate[i], W2) + b2, 0)

print(xW.shape)
#     if i % 100 == 0:
#         plt.plot(times, st[i], color='r', lw=2)
#         plt.plot(times, y[i], color='g', lw=1)
# plt.show()

# 0(1) ok
# 999(1000) ok
# 2999(3000) x
# 5009(5010) x
# 6002(6003) ok
# 7424(7425) x
# 9000(9001) x
# 10001(10002) ok
# 10029(10030) x
# plt.subplot(2, 1, i+1)

number = [i for i in range(b1.shape[0])]

# visualize xW
plt.plot(number, xW[0], color='r', lw=2)
plt.plot(number, xW[2999], color='g', lw=2)
plt.plot(number, xW[5009], color='b', lw=2)
plt.show()
plt.plot(number, xW[6002], color='r', lw=2)
plt.plot(number, xW[2999], color='g', lw=2)
plt.plot(number, xW[5009], color='b', lw=2)
plt.show()
plt.plot(number, xW[10001], color='r', lw=2)
plt.plot(number, xW[2999], color='g', lw=2)
plt.plot(number, xW[5009], color='b', lw=2)
plt.show()

plt.plot(times, st[0], color='r', lw=2)
plt.plot(times, st[2999], color='g', lw=2)
plt.plot(times, st[5009], color='b', lw=2)
plt.show()
plt.plot(times, st[6002], color='r', lw=2)
plt.plot(times, st[2999], color='g', lw=2)
plt.plot(times, st[5009], color='b', lw=2)
plt.show()
plt.plot(times, st[10001], color='r', lw=2)
plt.plot(times, st[2999], color='g', lw=2)
plt.plot(times, st[5009], color='b', lw=2)
plt.show()

# visualize xW
# plt.plot(number, xW[0], color='r', lw=2)
# plt.plot(number, xW[999], color='c', lw=2)
# plt.plot(number, xW[6002], color='g', lw=2)
# plt.plot(number, xW[2999], color='m', lw=2)
# plt.plot(number, xW[5009], color='b', lw=2)
# plt.show()

# visualize xW
# plt.plot(number, xW[0], color='r', lw=2)
# plt.plot(number, xW[999], color='c', lw=2)
# plt.plot(number, xW[6002], color='g', lw=2)
# plt.plot(number, xW[2999], color='m', lw=2)
# plt.plot(number, xW[5009], color='b', lw=2)
# plt.show()

# visualize xW
# plt.plot(number, xW[0], color='r', lw=2)
# plt.plot(number, xW[999], color='c', lw=2)
# plt.plot(number, xW[6002], color='g', lw=2)
# plt.plot(number, xW[2999], color='m', lw=2)
# plt.plot(number, xW[5009], color='b', lw=2)
# plt.show()
