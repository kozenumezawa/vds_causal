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

W12 = numpy.load('../npy/result_W12.npy')
b12 = numpy.load('../npy/result_b12.npy')
W45 = numpy.load('../npy/result_W45.npy')
b45 = numpy.load('../npy/result_b45.npy')

W23 = numpy.load('../npy/result_W23.npy')
b23 = numpy.load('../npy/result_b23.npy')
W34 = numpy.load('../npy/result_W34.npy')
b34 = numpy.load('../npy/result_b34.npy')

times = [i for i in range(st.shape[1])]
DATANUM = st.shape[0]

xW12 = numpy.zeros((DATANUM, b12.shape[0]), dtype=numpy.float32)
y12_ = numpy.zeros((DATANUM, b12.shape[0]), dtype=numpy.float32)
y12_activate = numpy.zeros((DATANUM, b12.shape[0]), dtype=numpy.float32)
y = numpy.zeros((DATANUM, b45.shape[0]), dtype=numpy.float32)

# visualize input and output (1st step)
for i in range(0, DATANUM):
    xW12[i] = numpy.matmul(st[i], W12)
    y12_[i] = xW12[i] + b12
    # y12_activate[i] = y12_[i] / (1 + numpy.absolute(y12_[i]))
    y12_activate[i] = y12_[i]
    y[i] = numpy.maximum(numpy.matmul(y12_activate[i], W45) + b45, 0)


# visualize input and output (2nd step)
xW23 = numpy.zeros((DATANUM, b23.shape[0]), dtype=numpy.float32)
y23_ = numpy.zeros((DATANUM, b23.shape[0]), dtype=numpy.float32)
y23_activate = numpy.zeros((DATANUM, b23.shape[0]), dtype=numpy.float32)
for i in range(0, DATANUM):
    xW23[i] = numpy.matmul(y12_activate[i], W23)

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


# visualize xW12
number = [i for i in range(b12.shape[0])]
plt.subplot(3, 1, 1)
plt.plot(number, xW12[0], color='r', lw=2)
plt.plot(number, xW12[2999], color='g', lw=2)
plt.plot(number, xW12[5009], color='b', lw=2)
plt.subplot(3, 1, 2)
plt.plot(number, xW12[6002], color='r', lw=2)
plt.plot(number, xW12[2999], color='g', lw=2)
plt.plot(number, xW12[5009], color='b', lw=2)
plt.subplot(3, 1, 3)
plt.plot(number, xW12[10001], color='r', lw=2)
plt.plot(number, xW12[2999], color='g', lw=2)
plt.plot(number, xW12[5009], color='b', lw=2)
plt.show()

# visualize xW23
number = [i for i in range(b23.shape[0])]
plt.subplot(6, 2, 1)
plt.plot(number, xW23[0], color='r', lw=2)
plt.plot(number, xW23[2999], color='g', lw=2)
plt.plot(number, xW23[5009], color='b', lw=2)
plt.subplot(6, 2, 3)
plt.plot(number, xW23[6002], color='r', lw=2)
plt.plot(number, xW23[2999], color='g', lw=2)
plt.plot(number, xW23[5009], color='b', lw=2)
plt.subplot(6, 2, 5)
plt.plot(number, xW23[10001], color='r', lw=2)
plt.plot(number, xW23[2999], color='g', lw=2)
plt.plot(number, xW23[5009], color='b', lw=2)

# visualize xW23
number = [i for i in range(b23.shape[0])]
plt.subplot(6, 2, 2)
plt.plot(number, xW23[999], color='r', lw=2)
plt.plot(number, xW23[7424], color='g', lw=2)
plt.plot(number, xW23[9000], color='b', lw=2)
plt.subplot(6, 2, 4)
plt.plot(number, xW23[6002], color='r', lw=2)
plt.plot(number, xW23[7424], color='g', lw=2)
plt.plot(number, xW23[9000], color='b', lw=2)
plt.subplot(6, 2, 6)
plt.plot(number, xW23[10001], color='r', lw=2)
plt.plot(number, xW23[7424], color='g', lw=2)
plt.plot(number, xW23[9000], color='b', lw=2)
plt.show()
