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

RS = 11

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
    n_unique_labels1 = unique_labels1.shape[0]
    n_unique_labels2 = unique_labels2.shape[0]
    n_samples1 = labels1.shape[0]
    n_samples2 = labels2.shape[0]
    if n_samples1 != n_samples2:
        raise Exception('oh no! n_samples are different lengths!')

    # Check that the two inputs have the same number of unique labels.
    if n_unique_labels1 != n_unique_labels2:
        print('oh no! unique labels are not the same length!')
        print('n_unique_labels1', n_unique_labels1)
        print('n_unique_labels2', n_unique_labels2)

    # Get the greater number of unique labels.
    greater_n_unique_labels = max(n_unique_labels1, n_unique_labels2)

    # Init empty masks.
    masks1 = np.zeros((greater_n_unique_labels, n_samples1), dtype=bool)
    masks2 = np.zeros((greater_n_unique_labels, n_samples2), dtype=bool)

    # Create an array for each unique value and add it to a mask.
    for i in range(n_unique_labels1):
        masks1[i] = np.array([label == unique_labels1[i] for label in labels1])
    for i in range(n_unique_labels2):
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

name = 'wine'
cluster_std = 10000
target = 'quality'
train = pd.read_csv(f'wine_train.csv')
test = pd.read_csv(f'wine_test.csv')
full = pd.concat([train, test])
y = np.array(train.loc[:,target])
X = np.array(train.drop(target, axis=1))

#name = 'blobs'
#cluster_std = 1
#X, y = make_blobs(
#    centers=6,
#    n_features=2,
#    n_samples=1000,
#    random_state=RS
#)

np.random.seed(RS)
random_states = np.random.choice(range(1000), size=5, replace=False)

metrics = {
    'MyKMeans': {'score': -1},
    'MyExpectMax': {'score': -1}
}

k_means_scores = []
expexct_max_scores = []
cluster_counts = []
total_start_time = time.time()
for n_clusters in range(2, np.unique(y).shape[0] + 2):
    best_score = {'MyKMeans': -1, 'MyExpectMax': -1}
    for random_state in random_states:
        clusterers = [
            MyKMeans(data=X, n_clusters=n_clusters, random_state=random_state),
            MyExpectMax(data=X, n_clusters=n_clusters, random_state=random_state, cluster_std=cluster_std)
        ]

        for clusterer in clusterers:
            alg = clusterer.__class__.__name__

            start_time = time.time()
            cluster_labels, centers, iters = clusterer.run()
            elapsed = round(time.time() - start_time, 3)
            score = silhouette_score(X, cluster_labels)

            if score > metrics[alg]['score'] or \
               (score == metrics[alg]['score'] and \
               (iters < metrics[alg]['iters'] or elapsed < metrics[alg]['elapsed'])):
                metrics[alg]['score'] = score
                metrics[alg]['n_clusters'] = n_clusters
                metrics[alg]['random_state'] = random_state
                metrics[alg]['iters'] = iters
                metrics[alg]['elapsed'] = elapsed
                metrics[alg]['cluster_std'] = cluster_std
                metrics[alg]['error'] = lowest_label_error(y, cluster_labels)
#            if score > best_score[alg]:
                best_score[alg] = score

    k_means_scores.append(best_score['MyKMeans'])
    expexct_max_scores.append(best_score['MyExpectMax'])
    cluster_counts.append(n_clusters)

print('k_means_scores:', k_means_scores)
print('expexct_max_scores:', expexct_max_scores)

pprint.PrettyPrinter(indent=4).pprint(metrics)
print('total_elapsed:', time.time() - total_start_time)

plot_stuff(cluster_counts, k_means_scores, expexct_max_scores, name)
