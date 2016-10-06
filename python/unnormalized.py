from __future__ import print_function

import httplib2

import numpy
import csv

# minS = numpy.min(S)
# maxS = numpy.max(S)
# minT = numpy.min(T)
# maxT = numpy.max(T)
# minX = numpy.min(X)
# maxX = numpy.max(X)
# S = (S - minS) / (maxS - minS)
# T = (T - minT) / (maxT - minT)
# X = (X - minX) / (maxX - minX)

rawdata = numpy.load('ocean.normalized.npy') # rawdata.shape = (10100, 2, 212)
s = numpy.zeros((rawdata.shape[0], rawdata.shape[2]), dtype=numpy.float32)
t = numpy.zeros((rawdata.shape[0], rawdata.shape[2]), dtype=numpy.float32)
v = numpy.zeros((rawdata.shape[0], rawdata.shape[2]), dtype=numpy.float32)

for i in range(rawdata.shape[0]):
    s[i, :rawdata.shape[2]] = rawdata[i, 0]          #   S(salinity)

maxS = 70
minS = 0.01
s = (maxS - minS) * s + minS

f = open('../csv/test.csv', 'w')
writer = csv.writer(f)
data = numpy.array(s, numpy.float32)
writer.writerows(data)
