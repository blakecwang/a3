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
        self.cluster_labels = np.zeros(len(self.data), dtype=int)
        self.prev_centroids = np.array([])
        self.iters = 0

    def run(self):
        self.__choose_random_centroids()
        while not np.array_equal(self.centroids, self.prev_centroids) and self.iters < self.max_iters:
            self.__assign_points_to_clusters()
            self.__find_new_centroids()
            self.iters += 1
        return self.cluster_labels, self.centroids, self.iters

    def __choose_random_centroids(self):
        np.random.seed(self.random_state)
        indices = np.random.choice(
            len(self.data),
            size=self.num_clusters,
            replace=False
        )
        self.centroids = self.data[indices]

    def __assign_points_to_clusters(self):
        for i, point in enumerate(self.data):
            shortest_distance = np.inf
            for j, centroid in enumerate(self.centroids):
                d = distance.euclidean(point, centroid)
                if d < shortest_distance:
                    self.cluster_labels[i] = j
                    shortest_distance = d

    def __find_new_centroids(self):
        self.prev_centroids = np.copy(self.centroids)
        for i in range(self.num_clusters):
            mask = np.array([label == i for label in self.cluster_labels])
            self.centroids[i] = np.mean(self.data[mask], axis=0)

# This is used both to when generating the data and when running KMeans.
NUM_CLUSTERS = 4

X, z = make_blobs(
    centers=NUM_CLUSTERS,
    n_samples=1000,
    cluster_std=0.99,
    random_state=12
)

clf = KMeans(data=X, num_clusters=NUM_CLUSTERS, max_iters=100)
cluster_labels, centroids, iters = clf.run()

print('iters:', iters)

plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, s=10)
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, marker='x')
plt.show()
