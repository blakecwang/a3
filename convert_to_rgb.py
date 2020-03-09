#!/usr/bin/env python3

import numpy as np

def convert_to_rgb(point):
    dim = len(point)
    if dim == 1:
        return np.array([255, point[0] * 255, 255], dtype=int)
    elif dim == 2:
        return np.array([point[0] * 255, 255, point[1] * 255], dtype=int)
    elif dim == 3:
        return np.array([x * 255 for x in point])
    else:
        if dim % 2 == 0:
            mid_chunk_len = 2 * round(dim / 6)
        else:
            mid_chunk_len = 2 * int(dim / 6) + 1
        end_chunk_len = round((dim - mid_chunk_len) / 2)
        p = np.zeros(3)
        p[0] = np.mean(point[:end_chunk_len])
        p[1] = np.mean(point[end_chunk_len:end_chunk_len + mid_chunk_len])
        p[2] = np.mean(point[end_chunk_len + mid_chunk_len:])
        return p
