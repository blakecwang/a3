#!/usr/bin/env python3

# https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html

import numpy as np
from scipy.stats import multivariate_normal

class ExpectMax:
    def __init__(self, data, num_clusters, random_state=11, max_iters=100, cluster_std=0.5):
        self.data = data
        self.num_clusters = num_clusters
        self.random_state = random_state
        self.max_iters = max_iters
        self.hidden_values = np.zeros((len(data), num_clusters))
        self.prev_means = np.array([])
        self.cluster_std = cluster_std
        self.cov = np.identity(data[0].shape[0]) * cluster_std
        self.iters = 0

    def run(self):
        self.__choose_random_means()
        while not np.array_equal(self.means, self.prev_means) and self.iters < self.max_iters:
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
        for i, mean in enumerate(self.means):
            distribution = multivariate_normal(mean, self.cov)
            for j, point in enumerate(self.data):
                self.hidden_values[j, i] = distribution.pdf(self.data[j])
        print(self.hidden_values)
        exit()

    def __find_new_means(self):
        self.prev_means = np.copy(self.means)
        for i in range(self.num_clusters):
            mask = np.array([label == i for label in self.hidden_values])
            self.means[i] = np.mean(self.data[mask], axis=0)
