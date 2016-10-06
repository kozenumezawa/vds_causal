# coding: utf-8

import numpy

data = numpy.load('../ocean.normalized.npy')

n = len(result[0]) // 2
st = numpy.zeros((data.shape[0], n * 2))
sv = numpy.zeros((data.shape[0], n * 2))
ts = numpy.zeros((data.shape[0], n * 2))
tv = numpy.zeros((data.shape[0], n * 2))
vs = numpy.zeros((data.shape[0], n * 2))
vt = numpy.zeros((data.shape[0], n * 2))
st[:, :n] = data[:, 0, :]
st[:, n:] = data[:, 1, :]
sv[:, :n] = data[:, 0, :]
sv[:, n:] = data[:, 2, :]
ts[:, :n] = data[:, 1, :]
ts[:, n:] = data[:, 0, :]
tv[:, :n] = data[:, 1, :]
tv[:, n:] = data[:, 2, :]
vs[:, :n] = data[:, 2, :]
vs[:, n:] = data[:, 0, :]
vt[:, :n] = data[:, 2, :]
vt[:, n:] = data[:, 1, :]


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
    y23_[i] = xW23[i] + b23
    y23_activate[i] = y23_[i] / (1 + numpy.absolute(y23_[i]))

xW56 = numpy.zeros((DATANUM, b56.shape[0]), dtype=numpy.float32)
y56_ = numpy.zeros((DATANUM, b56.shape[0]), dtype=numpy.float32)
y56_activate = numpy.zeros((DATANUM, b56.shape[0]), dtype=numpy.float32)
for i in range(0, DATANUM):
    xW56[i] = numpy.matmul(y23_activate[i], W56)

    #   write data to make cusal volume
    N = 26      #   focus on a 3rd neuron because it may react to causality

    print('sv')
    output = numpy.dot(sv, result[N])
    f = open('each_point_sv.csv', 'w')
    writer = csv.writer(f)
    data = [['lat', 'lon', 'value']]
    writer.writerows(data)

    lat = 30.0
    lon = 150.0
    for i, W in enumerate(sv):
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

        writer.writerows([[lat, lon, output[i]]])
        if i % 100 == 99:
            lat += 0.1
    f.close()
    print('max', numpy.max(output))
    print('min', numpy.min(output))


    print('tv')
    output = numpy.dot(tv, result[N])
    f = open('each_point_tv.csv', 'w')
    writer = csv.writer(f)
    data = [['lat', 'lon', 'value']]
    writer.writerows(data)

    lat = 30.0
    lon = 150.0
    for i, W in enumerate(tv):
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

        writer.writerows([[lat, lon, output[i]]])
        if i % 100 == 99:
            lat += 0.1
    f.close()
    print('max', numpy.max(output))
    print('min', numpy.min(output))



    print('vs')
    output = numpy.dot(vs, result[N])
    f = open('each_point_vs.csv', 'w')
    writer = csv.writer(f)
    data = [['lat', 'lon', 'value']]
    writer.writerows(data)

    lat = 30.0
    lon = 150.0
    for i, W in enumerate(sv):
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

        writer.writerows([[lat, lon, output[i]]])
        if i % 100 == 99:
            lat += 0.1
    f.close()
    print('max', numpy.max(output))
    print('min', numpy.min(output))
