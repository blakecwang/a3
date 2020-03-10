#!/usr/bin/env python3

# https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-centers.html

import numpy as np
from scipy.stats import multivariate_normal
from scipy.spatial import distance
from math import floor

class MyExpectMax:
    def __init__(self, data, n_clusters, random_state=11, max_iters=100, cluster_std=1.0, tol=0.1):
        self.data = data
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iters = max_iters
        self.hidden_values = np.zeros((len(data), n_clusters))
        self.prev_centers = np.zeros((n_clusters, data.shape[1]))
        self.cluster_std = cluster_std
        self.cov = np.identity(data.shape[1]) * cluster_std
        self.tol = tol
        self.iters = 0

    def run(self):
        self.__choose_random_centers()
        while self.__total_distance(self.centers, self.prev_centers) > self.tol and self.iters < self.max_iters:
            self.__calc_hidden_values()
            self.__find_new_centers()
            self.iters += 1
        cluster_labels = np.array([np.argmax(row) for row in self.hidden_values])
        return cluster_labels, self.centers, self.iters

    def __choose_random_centers(self):
        np.random.seed(self.random_state)
        indices = np.random.choice(
            self.data.shape[0],
            size=self.n_clusters,
            replace=False
        )
        self.centers = self.data[indices]

    def __calc_hidden_values(self):
        distributions = np.array(
            [multivariate_normal(center, self.cov) for center in self.centers]
        )
        for i, point in enumerate(self.data):
            for j, distribution in enumerate(distributions):
                self.hidden_values[i, j] = distribution.pdf(point)
            z_sum = np.sum(self.hidden_values[i])
            for j in range(self.n_clusters):
                self.hidden_values[i, j] = self.hidden_values[i, j] / z_sum

    def __find_new_centers(self):
        self.prev_centers = np.copy(self.centers)
        for i in range(self.n_clusters):
            self.centers[i] = np.average(self.data, axis=0, weights=self.hidden_values[:, i])

    def __total_distance(self, points1, points2):
        total = 0
        for i in range(points1.shape[0]):
            total += distance.euclidean(points1[i], points2[i])
        return total
