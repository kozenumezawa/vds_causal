from __future__ import print_function

import httplib2

import numpy

from pydap.client import open_url
import pydap.lib
from pydap.util import socks

pydap.lib.CACHE = '/tmp/pydap-cache/'
# pydap.lib.PROXY = httplib2.ProxyInfo(socks.PROXY_TYPE_HTTP,
#                                      'proxy.kuins.net', 8080)


#url = 'http://dias:akaika0530@dias-tb2.tkl.iis.u-tokyo.ac.jp:10080/'\
url = 'http://133.3.250.177/'\
        # 'thredds/dodsC/DIAS/MOVE-RA2014?'\
        'thredds/dodsC/fora-agg?'\
        'so[2122:2333][0][152:252][332:431],'\
        'to[2122:2333][0][152:252][332:431]'
        # 'S[2122:2333][0][152:252][332:431],'\
        # 'T[2122:2333][0][152:252][332:431],'\
        # 'U[2122:2333][0][152:252][332:431],'\
        # 'V[2122:2333][0][152:252][332:431],'\
        # 'W[2122:2333][0][152:252][332:431]'

dataset = open_url(url)

print(dataset.T.shape)
shape = dataset.T.shape

S = numpy.array(dataset.S.array[2122:2333 + 1, 0, 152:252 + 1, 332:431 + 1])
T = numpy.array(dataset.T.array[2122:2333 + 1, 0, 152:252 + 1, 332:431 + 1])
U = numpy.array(dataset.U.array[2122:2333 + 1, 0, 152:252 + 1, 332:431 + 1])
V = numpy.array(dataset.V.array[2122:2333 + 1, 0, 152:252 + 1, 332:431 + 1])
W = numpy.array(dataset.W.array[2122:2333 + 1, 0, 152:252 + 1, 332:431 + 1])
X = numpy.sqrt(U ** 2 + V ** 2 + W ** 2)

n = shape[2] * shape[3]
m = shape[0]
for i in range(shape[2]):
    for j in range(shape[3]):
        index = i * shape[3] + j
        result[index, 0, :] = S[:, 0, i, j]
        result[index, 1, :] = T[:, 0, i, j]
        result[index, 2, :] = X[:, 0, i, j]
numpy.save('ocean.npy', result)


minS = numpy.min(S)
maxS = numpy.max(S)
minT = numpy.min(T)
maxT = numpy.max(T)
minX = numpy.min(X)
maxX = numpy.max(X)
S = (S - minS) / (maxS - minS)
T = (T - minT) / (maxT - minT)
X = (X - minX) / (maxX - minX)

n = shape[2] * shape[3]
m = shape[0]
result = numpy.zeros((n, 3, m))
for i in range(shape[2]):
    for j in range(shape[3]):
        index = i * shape[3] + j
        result[index, 0, :] = S[:, 0, i, j]
        result[index, 1, :] = T[:, 0, i, j]
        result[index, 2, :] = X[:, 0, i, j]
numpy.save('normalized_ocean.npy', result)
