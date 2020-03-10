#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import silhouette_score
from classes.my_k_means import MyKMeans
from classes.my_expect_max import MyExpectMax

X, y = make_blobs(
    centers=3,
    n_features=10,
    n_samples=1000,
    random_state=11
)

for n_clusters in range(4,5):
    clusterers = [
        MyKMeans(data=X, n_clusters=n_clusters),
#        MyExpectMax(data=X, n_clusters=n_clusters)
    ]

    for clusterer in clusterers:
        print(n_clusters, clusterer.__class__.__name__)
        cluster_labels, centers, iters = clusterer.run()
#        score = silhouette_score(X, cluster_labels)
#        print(f"  {score}")
#        print('')

#        print('cluster_labels', np.unique(cluster_labels, return_counts=True))
#        print('centers', centers)
#        print('iters', iters)
