# coding: utf-8

import matplotlib.pyplot as plt
import numpy
import tensorflow as tf
import csv
import random

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

W56 = numpy.load('../npy/result_W56.npy')
b56 = numpy.load('../npy/result_b56.npy')
W67 = numpy.load('../npy/result_W67.npy')
b67 = numpy.load('../npy/result_b67.npy')

times = [i for i in range(st.shape[1])]
DATANUM = st.shape[0]

xW12 = numpy.zeros((DATANUM, b12.shape[0]), dtype=numpy.float32)
y12_ = numpy.zeros((DATANUM, b12.shape[0]), dtype=numpy.float32)
y12_activate = numpy.zeros((DATANUM, b12.shape[0]), dtype=numpy.float32)

# visualize input and output (1st step)
for i in range(0, DATANUM):
    xW12[i] = numpy.matmul(st[i], W12)
    y12_[i] = xW12[i] + b12
    # y12_activate[i] = y12_[i] / (1 + numpy.absolute(y12_[i]))
    y12_activate[i] = y12_[i]

# visualize input and output (2nd step)
xW23 = numpy.zeros((DATANUM, b23.shape[0]), dtype=numpy.float32)
y23_ = numpy.zeros((DATANUM, b23.shape[0]), dtype=numpy.float32)
y23_activate = numpy.zeros((DATANUM, b23.shape[0]), dtype=numpy.float32)
for i in range(0, DATANUM):
    xW23[i] = numpy.matmul(y12_activate[i], W23)
    y23_[i] = xW23[i] + b23
    y23_activate[i] = y23_[i] / (1 + numpy.absolute(y23_[i]))

# 3rd step
xW56 = numpy.zeros((DATANUM, b56.shape[0]), dtype=numpy.float32)
y56_ = numpy.zeros((DATANUM, b56.shape[0]), dtype=numpy.float32)
y56_activate = numpy.zeros((DATANUM, b56.shape[0]), dtype=numpy.float32)
for i in range(0, DATANUM):
    xW56[i] = numpy.matmul(y23_activate[i], W56)
    y56_[i] = xW56[i] + b56
    y56_activate[i] = y56_[i] / (1 + numpy.absolute(y56_[i]))

# 4th step
xW67 = numpy.zeros((DATANUM, b23.shape[0]), dtype=numpy.float32)
y67_ = numpy.zeros((DATANUM, b23.shape[0]), dtype=numpy.float32)
y67_activate = numpy.zeros((DATANUM, b23.shape[0]), dtype=numpy.float32)
for i in range(0, DATANUM):
    xW67[i] = numpy.matmul(y56_activate[i], W67)
    y67_[i] = xW67[i] + b67
    y67_activate[i] = y67_[i] / (1 + numpy.absolute(y67_[i]))

# 5th step
xW34 = numpy.zeros((DATANUM, b12.shape[0]), dtype=numpy.float32)
y34_ = numpy.zeros((DATANUM, b12.shape[0]), dtype=numpy.float32)
y34_activate = numpy.zeros((DATANUM, b12.shape[0]), dtype=numpy.float32)
for i in range(0, DATANUM):
    xW34[i] = numpy.matmul(y67_activate[i], W34)
    y34_[i] = xW34[i] + b34
    y34_activate[i] = y34_[i] / (1 + numpy.absolute(y34_[i]))

# 6th step
y = numpy.zeros((DATANUM, b45.shape[0]), dtype=numpy.float32)
for i in range(0, DATANUM):
    y[i] = numpy.maximum(numpy.matmul(y34_activate[i], W45) + b45, 0)

times = [i for i in range(PIXELS)]
# plt.plot(times, st[6002], color='r', lw=2)
# plt.show()
# plt.plot(times, y[6002], color='g', lw=1)
# plt.show()
# plt.plot(times, st[999], color='r', lw=2)
# plt.plot(times, y[999], color='g', lw=1)
# plt.show()
# plt.plot(times, st[0], color='r', lw=2)
# plt.plot(times, y[0], color='g', lw=1)
# plt.show()
# for i in range(0, DATANUM):
#     plt.plot(times, st[i], color='r', lw=2)
#     plt.plot(times, y[i], color='g', lw=1)
# plt.show()

# 0(1) ok
# 999(1000) ok
# 6002(6003) ok
# 10001(10002) ok
# 2(3) ok

# 2999(3000) x
# 5009(5010) x
# 7424(7425) x
# 9000(9001) x
# 10029(10030) x
# 7424(7425) x
# 2009(2010) x

# visualize xW56
number = [i for i in range(b56.shape[0])]
# plt.subplot(3, 1, 1)
# plt.plot(number, xW56[0], color='r', lw=2)
# plt.plot(number, xW56[2999], color='g', lw=2)
# plt.plot(number, xW56[5009], color='b', lw=2)
# plt.subplot(3, 1, 2)
# plt.plot(number, xW56[6002], color='r', lw=2)
# plt.plot(number, xW56[2999], color='g', lw=2)
# plt.plot(number, xW56[5009], color='b', lw=2)
# plt.subplot(3, 1, 3)
# plt.plot(number, xW56[10001], color='r', lw=2)
# plt.plot(number, xW56[2999], color='g', lw=2)
# plt.plot(number, xW56[5009], color='b', lw=2)
# plt.show()
# # for hitachi
# print(xW56[2][26])
# print(xW56[7424][26])
# print(xW56[2009][26])
# print(b12.shape[0])
# print(b23.shape[0])
# print(b56.shape[0])
#
# plt.subplot(3, 1, 1)
# plt.plot(number, xW56[999], color='r', lw=2)
# plt.plot(number, xW56[7424], color='g', lw=2)
# plt.plot(number, xW56[10029], color='b', lw=2)
# plt.subplot(3, 1, 2)
# plt.plot(number, xW56[6002], color='r', lw=2)
# plt.plot(number, xW56[2999], color='g', lw=2)
# plt.plot(number, xW56[5009], color='b', lw=2)
# plt.subplot(3, 1, 3)
# plt.plot(number, xW56[10001], color='r', lw=2)
# plt.plot(number, xW56[2999], color='g', lw=2)
# plt.plot(number, xW56[5009], color='b', lw=2)
# plt.show()

# search specific neuron output
y_ = numpy.zeros((DATANUM), dtype=numpy.float32)
for i in range(DATANUM):
    y_[i] = xW56[i][26]
print(numpy.argmax(y_))

#   write data to make cusal volume
N = 26
f = open('../csv/each_point_st.csv', 'w')
writer = csv.writer(f)
data = [['lat', 'lon', 'value']]
writer.writerows(data)

lat = 30.0
lon = 150.0
for i in range(DATANUM):
    if lat > 40.0:
        lat = 30.0
    if i % 100 == 98:
        lon = 158.7
    elif i % 100 == 99:
        lon = 160
    elif i % 100 == 0:
        lon = 150
    else:
        lon += 0.1

    writer.writerows([[lat, lon, xW56[i][N]]])
    if i % 100 == 99:
        lat += 0.1
f.close()
test = random.randint(0,10100)
print(test)
print(xW56[test][N])
print(xW56[0][N])
print(xW56[999][N])
print(xW56[6002][N])
print(xW56[10001][N])
print(xW56[2][N])
print(xW56[205][N])

print(xW56[2999][N])
print(xW56[5009][N])
print(xW56[7424][N])
print(xW56[9000][N])
print(xW56[10029][N])
print(xW56[2009][N])
print(xW56[9950][N])
print(xW56[9495][N])
