# coding: utf-8

import numpy
import csv

def main():
    data = numpy.load('../ocean.normalized.npy')
    result = numpy.load('../result.npy')
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


    #   write data to make cusal volume
    N = 3      #   focus on a 3rd neuron because it may react to causality

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

if __name__ == '__main__':
    main()
