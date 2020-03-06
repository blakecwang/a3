#!/usr/bin/env python3

# https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from scipy.spatial import distance

class KMeans:
    def __init__(self, data, num_clusters, random_state=11, max_iters=100):
        self.data = data
        self.num_clusters = num_clusters
        self.random_state = random_state
        self.max_iters = max_iters

    def run(self):
        self.__choose_random_centroids()

        prev_centroids = np.array([])
        cluster_labels = np.zeros(len(self.data), dtype=int)
        iters = 0
        while not np.array_equal(self.centroids, prev_centroids) and iters < self.max_iters:
            iters += 1
            for i, point in enumerate(self.data):
                shortest_distance = np.inf
                for j, centroid in enumerate(self.centroids):
                    d = distance.euclidean(point, centroid)
                    if d < shortest_distance:
                        cluster_labels[i] = j
                        shortest_distance = d

        return cluster_labels

        # Loop until the centroids don't change.

        #   Assign each data point to a cluster by min distance to centroid.
        #   Recalculate the centroid of each cluster.

    def __choose_random_centroids(self):
        np.random.seed(self.random_state)
        indices = np.random.choice(
            len(self.data),
            size=self.num_clusters,
            replace=False
        )
        self.centroids = self.data[indices]

NUM_CLUSTERS = 3
CLUSTER_STD = 0.8
RANDOM_STATE=12

X, z = make_blobs(
    n_samples=300,
    centers=NUM_CLUSTERS,
    cluster_std=CLUSTER_STD,
    random_state=RANDOM_STATE
)

clf = KMeans(data=X, num_clusters=NUM_CLUSTERS)
colors = clf.run()

plt.scatter(X[:, 0], X[:, 1], s=50, c=colors)
plt.show()
