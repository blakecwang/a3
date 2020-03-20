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
#    FastICA(random_state=RS),
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
    transform_start_time = time.time()
    if transformer is None:
        key = 'None'
        X_train_new = np.copy(X_train)
        X_test_new = np.copy(X_test)
    else:
        key = transformer.__class__.__name__
        transformer.fit(X_train)
        X_train_new = transformer.transform(X_train)
        X_test_new = transformer.transform(X_test)
    transform_time = round(time.time() - transform_start_time, 3)
    print(key)

    try:
        # Perform a grid seach to find the best hyper parameters.
        grid = {
            'random_state': [RS],
            'hidden_layer_sizes': [(100,)],
            'alpha': [0.0001, 0.0002, 0.0003, 0.0004],
            'learning_rate_init': [0.001, 0.002, 0.003, 0.004],
            'max_iter': [500]
        }
        clf = GridSearchCV(MLPClassifier(), grid, n_jobs=-1, cv=5)
        grid_start_time = time.time()
        clf.fit(X_train_new, y_train)
        grid_time = round(time.time() - grid_start_time, 3)

        # Write down the best validatoin score from grid search.
        metrics[key]['grid_cv_score'] = clf.cv_results_['mean_test_score']
        print(metrics[key]['grid_cv_score'])

        # Define new classifier based on best hyperparamters
        clf = MLPClassifier(**clf.best_params_)

        # Train the new classifier on all training data.
        train_start_time = time.time()
        train_time = round(time.time() - train_start_time, 3)

        metrics[key]['accuracy'] = accuracy
        metrics[key]['iters'] = iters
        metrics[key]['transform_time'] = transform_time
        metrics[key]['grid_time'] = grid_time
        metrics[key]['train_time'] = train_time
    except Exception as e:
        print('EXCEPTION!')
        print(e)

pprint.PrettyPrinter(indent=4).pprint(metrics)
print('total_time:', time.time() - total_start_time)