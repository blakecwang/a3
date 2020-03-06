#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs

# https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html

# I'm cheating here because I know the number of clusters ahead of time.

NUM_CLUSTERS = 3
CLUSTER_STD = 0.8
RANDOM_STATE=11

X, z = make_blobs(
    n_samples=300,
    centers=NUM_CLUSTERS,
    cluster_std=CLUSTER_STD,
    random_state=RANDOM_STATE
)

#plt.scatter(X[:, 0], X[:, 1], s=50)
#plt.show()

class KMeans:
    def __init__(self, num_clusters, cluster_std=0.5):
        self.num_clusters = num_clusters
        self.cluster_std = cluster_std

    def dump(self):
        print('num_clusters', self.num_clusters)
        print('cluster_std', self.cluster_std)

clf = KMeans(num_clusters=20, cluster_std=50)
clf.dump()
