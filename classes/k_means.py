#!/usr/bin/env python3

# https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html

import numpy as np
from scipy.spatial import distance

class KMeans:
    def __init__(self, data, num_clusters, random_state=11, max_iters=100):
        self.data = data
        self.num_clusters = num_clusters
        self.random_state = random_state
        self.max_iters = max_iters
        self.cluster_labels = np.zeros(len(data), dtype=int)
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
