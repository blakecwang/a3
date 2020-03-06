#!/usr/bin/env python3

# https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from scipy.spatial import distance

class KMeans:
    def __init__(self, data, num_clusters, random_state):
        self.data = data
        self.num_clusters = num_clusters
        self.random_state = random_state

    def run(self):
#        distance.euclidean([1, 0, 0], [0, 1, 0])

        # Initially choose centroids to be K random points from the data.
        self.__choose_random_centroids()
        print(self.centroids)

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
RANDOM_STATE=11

X, z = make_blobs(
    n_samples=300,
    centers=NUM_CLUSTERS,
    cluster_std=CLUSTER_STD,
    random_state=RANDOM_STATE
)

#plt.scatter(X[:, 0], X[:, 1], s=50)
#plt.show()

clf = KMeans(data=X, num_clusters=NUM_CLUSTERS, random_state=RANDOM_STATE)
clf.run()
