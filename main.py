#!/usr/bin/env python3

import numpy as np
import time
import pprint
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import silhouette_score
from classes.my_k_means import MyKMeans
from classes.my_expect_max import MyExpectMax

X, y = make_blobs(
    centers=3,
    n_features=2,
    n_samples=1000,
    random_state=11
)

np.random.seed(11)
random_states = np.random.choice(range(1000), size=5, replace=False)

metrics = {
    'MyKMeans': {'score': -1},
    'MyExpectMax': {'score': -1}
}

pp = pprint.PrettyPrinter(indent=4)

for n_clusters in range(3, 8):
    for random_state in random_states:
        clusterers = [
            MyKMeans(data=X, n_clusters=n_clusters, random_state=random_state),
            MyExpectMax(data=X, n_clusters=n_clusters, random_state=random_state)
        ]

        for clusterer in clusterers:
            start_time = time.time()
            cluster_labels, centers, iters = clusterer.run()
            elapsed = round(time.time() - start_time, 3)
            score = silhouette_score(X, cluster_labels)

            alg = clusterer.__class__.__name__
            if score > metrics[alg]['score'] or \
               (score == metrics[alg]['score'] and \
               (iters < metrics[alg]['iters'] or elapsed < metrics[alg]['elapsed'])):
                metrics[alg]['score'] = score
                metrics[alg]['n_clusters'] = n_clusters
                metrics[alg]['random_state'] = random_state
                metrics[alg]['iters'] = iters
                metrics[alg]['elapsed'] = elapsed

pp.pprint(metrics)
