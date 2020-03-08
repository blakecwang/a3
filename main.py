#!/usr/bin/env python3

import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from classes.k_means import KMeans
from classes.expect_max import ExpectMax

# These are used both for generating data and for calculating clusters (kind of
# cheating).
NUM_CLUSTERS = 4
CLUSTER_STD = 0.9

X, z = make_blobs(
    centers=NUM_CLUSTERS,
    cluster_std=CLUSTER_STD,
    n_samples=1000,
    random_state=12
)

#k_means = KMeans(data=X, num_clusters=NUM_CLUSTERS)
#hidden_values, centroids, iters = k_means.run()
#
#print('iters:', iters)
#
#plt.scatter(X[:, 0], X[:, 1], c=hidden_values, s=10)
#plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, marker='x')
#plt.savefig(f"k_means_clusters.png")
##plt.show()



expect_max = ExpectMax(data=X, num_clusters=NUM_CLUSTERS, cluster_std=CLUSTER_STD)
expect_max.run()

#hidden_values, centroids, iters = expect_max.run()
#
#print('iters:', iters)
#
#plt.scatter(X[:, 0], X[:, 1], c=hidden_values, s=10)
#plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, marker='x')
#plt.savefig(f"k_means_clusters.png")
##plt.show()
