import json

import numpy

def split(item):
    reshaped = item.reshape((2, len(item) // 2))
    return {
        'x': [str(v) for v in reshaped[0]],
        'y': [str(v) for v in reshaped[1]],
    }

def main():
    data = numpy.load('ocean.normalized.npy')
    result = numpy.load('result.npy')
    obj = {
        'data': [{'s': list(item[0]), 't': list(item[1]), 'v': list(item[2])} for item in data],
        'features': [split(item) for item in result],
    }
    json.dump(obj, open('data.json', 'w'), sort_keys=True, indent=2)

if __name__ == '__main__':
    main()
