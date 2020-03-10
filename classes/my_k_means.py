#!/usr/bin/env python3

# https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html

import numpy as np
from scipy.spatial import distance

class MyKMeans:
    def __init__(self, data, n_clusters, random_state=11, max_iters=100):
        self.data = data
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iters = max_iters
        self.cluster_labels = np.zeros(data.shape[0], dtype=int)
        self.prev_centers = np.zeros((n_clusters, data.shape[1]))
        self.iters = 0

    def run(self):
        self.__choose_random_centers()
        while not np.array_equal(self.centers, self.prev_centers) and self.iters < self.max_iters:
            self.__assign_points_to_clusters()
            self.__find_new_centers()
            self.iters += 1
        return self.cluster_labels, self.centers, self.iters

    def __choose_random_centers(self):
        np.random.seed(self.random_state)
        indices = np.random.choice(
            self.data.shape[0],
            size=self.n_clusters,
            replace=False
        )
        self.centers = self.data[indices]

    def __assign_points_to_clusters(self):
        for i, point in enumerate(self.data):
            shortest_distance = np.inf
            for j, center in enumerate(self.centers):
                d = distance.euclidean(point, center)
                if d < shortest_distance:
                    self.cluster_labels[i] = j
                    shortest_distance = d

    def __find_new_centers(self):
        self.prev_centers = np.copy(self.centers)
        for i in range(self.n_clusters):
            mask = np.array([label == i for label in self.cluster_labels])
            exit()
            if np.count_nonzero(mask) == 0:
                # If there are no points in this cluster, then set the cluster
                # center to be the mean of the other cluster centers.
                self.centers[i] = np.mean(self.centers, axis=0)
            else:
                self.centers[i] = np.mean(self.data[mask], axis=0)
