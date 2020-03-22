#!/usr/bin/env python3

import numpy as np
from scipy.spatial import distance

def label(centers, data):
    if centers.shape[1] != data.shape[1]:
        raise Exception('centers and data have different dimension!')
    ret = np.zeros(data.shape[0], dtype=int)
    for i, point in enumerate(data):
        shortest_d = np.inf
        nearest_center_idx = -1
        for j, center in enumerate(centers):
            d = distance.euclidean(point, center)
            if d < shortest_d:
                shortest_d = d
                ret[i] = j
    return ret

centers = np.array([[0,0], [5,5], [1.5,1.5]])
data = np.array([[0,0], [1,1], [2,2]])
ret = label(centers, data)
print(ret)
