#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from classes.k_means import KMeans
from classes.expect_max import ExpectMax

# These are used both for generating data and for calculating clusters (kind of
# cheating).
NUM_CLUSTERS = 4
CLUSTER_STD = 0.9

X, z = make_blobs(
    centers=NUM_CLUSTERS,
    cluster_std=CLUSTER_STD,
    n_samples=1000,
    random_state=12
)


#k_means = KMeans(data=X, num_clusters=NUM_CLUSTERS)
#hidden_values, centroids, iters = k_means.run()
#
#print('iters:', iters)
#
#plt.scatter(X[:, 0], X[:, 1], c=hidden_values, s=10)
#plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, marker='x')
#plt.savefig(f"k_means_clusters.png")
##plt.show()

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

expect_max = ExpectMax(data=X, num_clusters=NUM_CLUSTERS, cluster_std=CLUSTER_STD, max_iters=1)
hidden_values, means, iters = expect_max.run()
colors = np.array([convert_to_rgb(val) for val in hidden_values])

print('iters:', iters)

plt.scatter(X[:, 0], X[:, 1], c=colors, s=10)
plt.scatter(means[:, 0], means[:, 1], c='black', s=200, marker='x')
#plt.savefig(f"expect_max_clusters.png")
plt.show()
