#!/usr/bin/env python3

# https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html

import numpy as np
from scipy.stats import multivariate_normal
from scipy.spatial import distance
from math import floor

class ExpectMax:
    def __init__(self, data, num_clusters, random_state=11, max_iters=100, cluster_std=0.5, tol=0.1):
        self.data = data
        self.num_clusters = num_clusters
        self.random_state = random_state
        self.max_iters = max_iters
        self.hidden_values = np.zeros((len(data), num_clusters))
        self.prev_means = np.zeros((num_clusters, data.shape[1]))
        self.cluster_std = cluster_std
        self.cov = np.identity(data[0].shape[0]) * cluster_std
        self.tol = 0.1
        self.iters = 0

    def run(self):
        self.__choose_random_means()
        while self.__total_distance(self.means, self.prev_means) > self.tol and self.iters < self.max_iters:
            self.__calc_hidden_values()
            self.__find_new_means()
            self.iters += 1
        return self.hidden_values, self.means, self.iters

    def __choose_random_means(self):
        np.random.seed(self.random_state)
        indices = np.random.choice(
            len(self.data),
            size=self.num_clusters,
            replace=False
        )
        self.means = self.data[indices]

    def __calc_hidden_values(self):
        distributions = np.array(
            [multivariate_normal(mean, self.cov) for mean in self.means]
        )
        for i, point in enumerate(self.data):
            for j, distribution in enumerate(distributions):
                self.hidden_values[i, j] = distribution.pdf(point)
            z_sum = np.sum(self.hidden_values[i])
            for j in range(self.num_clusters):
                self.hidden_values[i, j] = self.hidden_values[i, j] / z_sum

    def __find_new_means(self):
        self.prev_means = np.copy(self.means)
        for i in range(self.num_clusters):
            self.means[i] = np.average(self.data, axis=0, weights=self.hidden_values[:, i])

    def __total_distance(self, before, after):
        total = 0
        for i in range(before.shape[0]):
            total += distance.euclidean(before[i], after[i])
        return total
