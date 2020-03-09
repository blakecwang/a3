#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from classes.k_means import KMeans
from classes.expect_max import ExpectMax
from classes.helpers import Helpers

# These are used both for generating data and for calculating clusters (kind of
# cheating).
NUM_CLUSTERS = 3
CLUSTER_STD = 1

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

expect_max = ExpectMax(data=X, num_clusters=NUM_CLUSTERS, cluster_std=CLUSTER_STD, max_iters=1)
hidden_values, means, iters = expect_max.run()
colors = np.array([np.argmax(v_row) for v_row in hidden_values])

print('iters:', iters)

plt.scatter(X[:, 0], X[:, 1], c=colors, s=10)
plt.scatter(means[:, 0], means[:, 1], c='black', s=200, marker='x')
#plt.savefig(f"expect_max_clusters.png")
plt.show()
