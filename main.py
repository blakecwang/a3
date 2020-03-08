#!/usr/bin/env python3

import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from classes.k_means import KMeans

# This is used both to when generating the data and when running KMeans.
NUM_CLUSTERS = 4

X, z = make_blobs(
    centers=NUM_CLUSTERS,
    cluster_std=0.9,
    n_samples=1000,
    random_state=12
)

clf = KMeans(data=X, num_clusters=NUM_CLUSTERS)
cluster_labels, centroids, iters = clf.run()

print('iters:', iters)

plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, s=10)
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, marker='x')
plt.show()
