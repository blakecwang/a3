#!/usr/bin/env python3

import numpy as np
import time
import string
import pprint
import itertools
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from classes.my_k_means import MyKMeans
from classes.my_expect_max import MyExpectMax

dataset_attrs = [('wine', 'quality'), ('breast_cancer', 'Diagnosis')]

#for filename, target in dataset_attrs:
#
#    train = pd.read_csv(f'{filename}_train.csv')
#    test = pd.read_csv(f'{filename}_test.csv')
#
#    y_train = train.loc[:,target]
#    X_train = train.drop(target, axis=1)
#    y_test = test.loc[:,target]
#    X_test = test.drop(target, axis=1)

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

def lowest_label_error(labels1, labels2):
    n_labels = np.unique(labels1).shape[0]
    masks1 = np.zeros((n_labels, labels1.shape[0]), dtype=bool)
    masks2 = np.copy(masks1)
    for i in range(n_labels):
        masks1[i] = np.array([label == i for label in labels1])
    for i in range(n_labels):
        masks2[i] = np.array([label == i for label in labels2])

    lowest_error = np.inf
    for masks2_perm in itertools.permutations(masks2):
        error = np.count_nonzero(masks1 != masks2_perm)
        if error < lowest_error:
            lowest_error = error
    return lowest_error

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
                metrics[alg]['error'] = lowest_label_error(cluster_labels, y)

pp.pprint(metrics)

#{   'MyExpectMax': {   'elapsed': 0.228,
#                       'iters': 3,
#                       'n_clusters': 3,
#                       'random_state': 730,
#                       'score': 0.698220165940078},
#    'MyKMeans': {   'elapsed': 0.114,
#                    'iters': 3,
#                    'n_clusters': 3,
#                    'random_state': 730,
#                    'score': 0.698220165940078}}
