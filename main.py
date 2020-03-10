#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from classes.my_k_means import MyKMeans
from classes.my_expect_max import MyExpectMax

N_CLUSTERS = 5

X, y = make_blobs(
    centers=N_CLUSTERS,
    n_features=3,
    n_samples=1000,
    random_state=11
)

clusterers = [
    MyKMeans(data=X, n_clusters=N_CLUSTERS),
    MyExpectMax(data=X, n_clusters=N_CLUSTERS)
]

for clusterer in clusterers:
    cluster_labels, centers, iters = clusterer.run()
    print('cluster_labels', np.unique(cluster_labels, return_counts=True))
    print('centers', centers)
    print('iters', iters)
