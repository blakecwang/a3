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
    n_labels = np.unique(labels1).shape[0]
    masks1 = np.zeros((n_labels, labels1.shape[0]), dtype=bool)
    masks2 = np.copy(masks1)
    for i in range(n_labels):
        masks1[i] = np.array([label == i for label in labels1])
    for i in range(n_labels):
        masks2[i] = np.array([label == i for label in labels2])

    lowest_error = np.inf
    for masks2_perm in itertools.permutations(masks2):
        masks2_perm = np.array(masks2_perm)
        error = 0
        for i in range(masks1.shape[0]):
            if not np.array_equal(masks1[i], masks2_perm[i]):
                error += 1
        if error < lowest_error:
            lowest_error = error
    return lowest_error / labels1.shape[0]

#target = 'quality'
#train = pd.read_csv(f'wine_train.csv')
#test = pd.read_csv(f'wine_test.csv')
#full = pd.concat([train, test])
#y = np.array(train.loc[:,target])
#X = np.array(train.drop(target, axis=1))

X, y = make_blobs(
    centers=6,
    n_features=2,
    n_samples=1000,
    random_state=11
)
print('score:', silhouette_score(X, y))

np.random.seed(11)
random_states = np.random.choice(range(1000), size=5, replace=False)
#random_states =  [25]
random_states =  [11]

metrics = {
    'MyKMeans': {'score': -1},
    'MyExpectMax': {'score': -1}
}

k_means_scores = []
expexct_max_scores = []
cluster_counts = []
total_start_time = time.time()
for n_clusters in range(6,7):
    best_score = {'MyKMeans': -1, 'MyExpectMax': -1}
    for cluster_std in [10000]:
        for random_state in random_states:
            clusterers = [
#                MyKMeans(data=X, n_clusters=n_clusters, random_state=random_state),
                MyExpectMax(data=X, n_clusters=n_clusters, random_state=random_state, cluster_std=cluster_std)
            ]

            for clusterer in clusterers:
                alg = clusterer.__class__.__name__

                start_time = time.time()
                cluster_labels, centers, iters = clusterer.run()
                print('n_clusters', np.unique(n_clusters))
                print('cluster_labels', np.unique(cluster_labels))
                exit()
                elapsed = round(time.time() - start_time, 3)
                score = silhouette_score(X, cluster_labels)

#                scatter_stuff(X, cluster_labels, centers, alg)

                if score > metrics[alg]['score'] or \
                   (score == metrics[alg]['score'] and \
                   (iters < metrics[alg]['iters'] or elapsed < metrics[alg]['elapsed'])):
                    metrics[alg]['score'] = score
                    metrics[alg]['n_clusters'] = n_clusters
                    metrics[alg]['random_state'] = random_state
                    metrics[alg]['iters'] = iters
                    metrics[alg]['elapsed'] = elapsed
                    metrics[alg]['cluster_std'] = cluster_std
    #                metrics[alg]['error'] = lowest_label_error(cluster_labels, y)

                if score > best_score[alg]:
                    best_score[alg] = score

        k_means_scores.append(best_score['MyKMeans'])
        expexct_max_scores.append(best_score['MyExpectMax'])
        cluster_counts.append(n_clusters)

print('k_means_scores:', k_means_scores)
print('expexct_max_scores:', expexct_max_scores)

pprint.PrettyPrinter(indent=4).pprint(metrics)
print('total_elapsed:', time.time() - total_start_time)

plot_stuff(cluster_counts, k_means_scores, expexct_max_scores, 'wine')
