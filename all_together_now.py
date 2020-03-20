#!/usr/bin/env python3

import numpy as np
import time
import string
import pprint
import itertools
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from classes.my_k_means import MyKMeans
from classes.my_expect_max import MyExpectMax
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection

def scatter_stuff(X, y, centers, name):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=10)
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, marker='x')
    plt.xlabel('x1', fontsize=18, fontname='Arial')
    plt.ylabel('x2', fontsize=18, fontname='Arial')
    plt.tight_layout()
    plt.savefig(f"{name}_scatter.png")
#    plt.show()

def plot_stuff(cluster_counts, k_means, expect_max, name):
    font = { 'family': 'Arial', 'size': 18 }
    plt.rc('font', **font)
    plt.plot(cluster_counts, k_means, label='MyKMeans')
    plt.plot(cluster_counts, expect_max, label='MyExpectMax')
    plt.ylabel('Average Silhouette', fontsize=18, fontname='Arial')
    plt.xlabel('Clusters', fontsize=18, fontname='Arial')
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"{name}.png")
#    plt.show()

def lowest_label_error(labels1, labels2):
    # Get arrays of unique labels.
    unique_labels1 = np.unique(labels1)
    unique_labels2 = np.unique(labels2)

    # Check that the two inputs have the same number of unique labels.
    n_labels = unique_labels1.shape[0]
    if n_labels is not unique_labels2.shape[0]:
        print('unique_labels1', unique_labels1)
        print('unique_labels2', unique_labels2)
        raise Exception('oh no! labels are not the same length!')

    # Init empty masks.
    masks1 = np.zeros((n_labels, labels1.shape[0]), dtype=bool)
    masks2 = np.copy(masks1)

    # Create an array for each unique value and add it to a mask.
    for i in range(n_labels):
        masks1[i] = np.array([label == unique_labels1[i] for label in labels1])
    for i in range(n_labels):
        masks2[i] = np.array([label == unique_labels2[i] for label in labels2])

    # Find the lowest error between mask1 and every permutation of mask2.
    lowest_error = np.inf
    for masks2_perm in itertools.permutations(masks2):
        masks2_perm = np.array(masks2_perm)
        error = np.count_nonzero(masks1 != masks2_perm)
        if error < lowest_error:
            lowest_error = error

    # Return the error percentage.
    return lowest_error / masks1.size

RS = 11

## Wine Quality
#name = 'wq'
#cluster_std = 10000
#target = 'quality'
#train = pd.read_csv(f'wine_train.csv')
#test = pd.read_csv(f'wine_test.csv')
#full = pd.concat([train, test])
#y = np.array(train.loc[:,target])
#X = np.array(train.drop(target, axis=1))
#transformers = [
#    PCA(n_components=1, random_state=RS),
#    FastICA(random_state=RS),
#    GaussianRandomProjection(random_state=RS, n_components=9),
#    TruncatedSVD(n_components=1, random_state=RS)
#]

# Generated Blobs
name = 'gb'
cluster_std = 1
X, y = make_blobs(
    centers=6,
    n_features=2,
    n_samples=1000,
    random_state=11
)
transformers = [
    PCA(n_components=1, random_state=RS),
#    FastICA(random_state=RS),
#    GaussianRandomProjection(random_state=RS, n_components=1),
#    TruncatedSVD(n_components=1, random_state=RS)
]

np.random.seed(RS)
random_states = np.random.choice(range(1000), size=5, replace=False)
total_start_time = time.time()
n_clusters = np.unique(y).shape[0]
metrics = {
    'PCA_MyKMeans': {'best_score': -1},
    'PCA_MyExpectMax': {'best_score': -1},
    'FastICA_MyKMeans': {'best_score': -1},
    'FastICA_MyExpectMax': {'best_score': -1},
    'GaussianRandomProjection_MyKMeans': {'best_score': -1},
    'GaussianRandomProjection_MyExpectMax': {'best_score': -1},
    'TruncatedSVD_MyKMeans': {'best_score': -1},
    'TruncatedSVD_MyExpectMax': {'best_score': -1}
}
clusterers = [
#    'MyKMeans',
    'MyExpectMax'
]

for transformer in transformers:
    tf_name = transformer.__class__.__name__
#    X_new = transformer.fit_transform(X)
    X_new = np.copy(X)

    for clusterer in clusterers:
        try:
            key = f'{tf_name}_{clusterer}'
            print(key)

            for rs in random_states:
                print('rs', rs)
                # hack for weird behavior
                if name is 'wq' and key is 'FastICA_MyExpectMax' and rs is not 730:
                    continue

                if clusterer is 'MyKMeans':
                    clusterer = MyKMeans(data=X_new, n_clusters=n_clusters, random_state=rs)
                else:
                    clusterer = MyExpectMax(data=X_new, n_clusters=n_clusters, random_state=rs, cluster_std=10000)

                start_time = time.time()
                cluster_labels, centers, iters = clusterer.run()
                elapsed = round(time.time() - start_time, 3)
                score = silhouette_score(X_new, cluster_labels)

                if score > metrics[key]['best_score'] or \
                   (score == metrics[key]['best_score'] and \
                   (iters < metrics[key]['iters'] or elapsed < metrics[key]['elapsed'])):
                    metrics[key]['best_score'] = score
                    metrics[key]['n_clusters'] = n_clusters
                    metrics[key]['random_state'] = rs
                    metrics[key]['iters'] = iters
                    metrics[key]['elapsed'] = elapsed
                    metrics[key]['cluster_std'] = cluster_std
#                    metrics[key]['error'] = lowest_label_error(cluster_labels, y)
                    scatter_stuff(X_new, cluster_labels, centers, 'name')
        except Exception as e:
            print('EXCEPTION!')
            print(e)

pprint.PrettyPrinter(indent=4).pprint(metrics)
print('total_elapsed:', time.time() - total_start_time)
