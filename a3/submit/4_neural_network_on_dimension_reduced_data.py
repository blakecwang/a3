#!/usr/bin/env python3

import numpy as np
import time
import pprint
import itertools
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
#from sklearn.model_selection import cross_val_score

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


RS = 11

# Wine Quality
name = 'wq'
target = 'quality'
train = pd.read_csv(f'wine_train.csv')
test = pd.read_csv(f'wine_test.csv')

y_train = train.loc[:,target]
X_train = train.drop(target, axis=1)
y_test = test.loc[:,target]
X_test = test.drop(target, axis=1)

transformers = [
    None,
    PCA(n_components=1, random_state=RS),
    FastICA(random_state=RS),
    GaussianRandomProjection(random_state=RS, n_components=9),
    TruncatedSVD(n_components=1, random_state=RS)
]

np.random.seed(RS)
random_states = np.random.choice(range(1000), size=5, replace=False)
total_start_time = time.time()
metrics = {
    'None': {},
    'PCA': {},
    'FastICA': {},
    'GaussianRandomProjection': {},
    'TruncatedSVD': {}
}

for transformer in transformers:
    # Transform the data (or not).
    if transformer is None:
        key = 'None'
        X_train_new = np.copy(X_train)
        X_test_new = np.copy(X_test)
    else:
        key = transformer.__class__.__name__
        transform_start_time = time.time()
        transformer.fit(X_train)
        X_train_new = transformer.transform(X_train)
        X_test_new = transformer.transform(X_test)
        metrics[key]['transform_time'] = round(time.time() - transform_start_time, 3)
    print(key)

    try:
        # Perform a grid seach to find the best hyper parameters.
        grid = {
            'random_state': [RS],
            'hidden_layer_sizes': [(5,), (10,), (50,), (100,)],
            'alpha': [0.0001, 0.0002, 0.0003, 0.0004],
            'learning_rate_init': [1e-4, 1e-3, 1e-2],
            'max_iter': [500]
        }
        clf = GridSearchCV(MLPClassifier(), grid, n_jobs=-1, cv=5)
        grid_start_time = time.time()
        clf.fit(X_train_new, y_train)
        metrics[key]['grid_time'] = round(time.time() - grid_start_time, 3)
        metrics[key]['best_params'] = clf.best_params_

        # Define new classifier based on best hyperparamters
        clf = MLPClassifier(**clf.best_params_)

        # Train the new classifier on all training data.
        train_start_time = time.time()
        clf.fit(X_train_new, y_train)
        metrics[key]['iters'] = clf.n_iter_
        metrics[key]['train_time'] = round(time.time() - train_start_time, 3)

        # Calculate the final scores.
        metrics[key]['final_train_score'] = clf.score(X_train_new, y_train)
        metrics[key]['final_test_score'] = clf.score(X_test_new, y_test)
    except Exception as e:
        print('EXCEPTION!')
        print(e)

pprint.PrettyPrinter(indent=4).pprint(metrics)
print('total_time:', time.time() - total_start_time)
