#!/usr/bin/env python3

#import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from classes.my_k_means import MyKMeans
from classes.my_expect_max import MyExpectMax

N_CLUSTERS = 5
CLUSTER_STD = 1

X, y = make_blobs(
    centers=N_CLUSTERS,
    cluster_std=CLUSTER_STD,
    n_features=3,
    n_samples=1000,
    random_state=12
)

clusterer = MyKMeans(data=X, n_clusters=N_CLUSTERS)
cluster_labels, centers, iters = clusterer.run()
print('iters:', iters)
print('centers:', centers)

#plt.scatter(X[:, 0], X[:, 1], c=hidden_values, s=10)
#plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, marker='x')
#plt.savefig(f"my_k_means_clusters.png")
##plt.show()

clusterer = MyExpectMax(data=X, n_clusters=N_CLUSTERS, cluster_std=CLUSTER_STD)
colors, hidden_values, centers, iters = clusterer.run()
print('iters:', iters)
print('centers:', centers)

#plt.scatter(X[:, 0], X[:, 1], c=colors, s=10)
#plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, marker='x')
##plt.savefig(f"clusterer_clusters.png")
#plt.show()
