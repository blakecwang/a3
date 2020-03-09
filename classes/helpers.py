#!/usr/bin/env python3

import numpy as np

class Helpers:
    # This currently doesn't work for < 4 clusters.
    def convert_to_rgb(points):
        ret = np.zeros((len(points), 3), dtype=int)

        print(ret.__class__)
        print(ret.shape)
        print(ret[0])
        print(ret[0,0].__class__)

        for i, point in enumerate(points):
            dim = len(point)
            if dim == 1:
                ret[i] = np.array([
                    255,
                    round(point[0] * 255),
                    255
                ], dtype=int)
            elif dim == 2:
                ret[i] = np.array([
                    point[0] * 255,
                    255,
                    point[1] * 255
                ], dtype=int)
            elif dim == 3:
                ret[i] = np.array([round(x * 255) for x in point], dtype=int)
            else:
                if dim % 2 == 0:
                    mid_chunk_len = 2 * round(dim / 6)
                else:
                    mid_chunk_len = 2 * int(dim / 6) + 1
                end_chunk_len = round((dim - mid_chunk_len) / 2)

                ret[i, 0] = np.mean(point[:end_chunk_len])
                ret[i, 1] = np.mean(point[end_chunk_len:end_chunk_len + mid_chunk_len])
                ret[i, 2] = np.mean(point[end_chunk_len + mid_chunk_len:])
        print('-')
        print(ret.__class__)
        print(ret.shape)
        print(ret[0])
        print(ret[0,0].__class__)
        return ret
