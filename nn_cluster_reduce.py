#!/usr/bin/env python3

import numpy as np
import time
import pprint
import itertools
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from classes.my_k_means import MyKMeans
from classes.my_expect_max import MyExpectMax
from sklearn.metrics import silhouette_score

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
cluster_std = 10000
target = 'quality'
train = pd.read_csv(f'wine_train.csv')
test = pd.read_csv(f'wine_test.csv')

y_train = np.array(train.loc[:,target])
X_train = np.array(train.drop(target, axis=1))
y_test = np.array(test.loc[:,target])
X_test = np.array(test.drop(target, axis=1))

results = {
#    'Raw': {'score': -1},
    'MyKMeans': {'score': -1},
#    'MyExpectMax': {'score': -1}
}

np.random.seed(RS)
random_states = np.random.choice(range(1000), size=5, replace=False)
n_clusters = np.unique(y_train).shape[0]
total_start_time = time.time()
for key in results.keys():
    print(key)

    # Transform the data (or not).
    if key is 'Raw':
        # Just make a copy.
        X_train_new = np.copy(X_train)
        X_test_new = np.copy(X_test)
    else:
        X_train_new = None
        best_centers = None
        cluster_start_time = time.time()
        for rs in random_states:
            if key == 'MyKMeans':
                clusterer = MyKMeans(data=X_train, n_clusters=n_clusters, random_state=rs)
            elif key == 'MyExpectMax':
                clusterer = MyExpectMax(data=X_train, n_clusters=n_clusters, random_state=rs, cluster_std=cluster_std)
            cluster_labels, centers, iters = clusterer.run()
            score = silhouette_score(X_train, cluster_labels)
            if score > results[key]['score']:
                results[key]['score'] = score
                results[key]['cluster_iters'] = iters
                results[key]['n_unique_labels'] = np.unique(cluster_labels).shape[0]

                # Store the centers in order to be able to group test data into the same clustering.
                best_centers = centers

                # The cluster labels are the new training data.
                X_train_new = np.array(list(map(lambda cl: [cl], cluster_labels)))
        results[key]['cluster_time'] = time.time() - cluster_start_time

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
        results[key]['grid_time'] = round(time.time() - grid_start_time, 3)
        results[key]['best_params'] = clf.best_params_

        # Define new classifier based on best hyperparamters
        clf = MLPClassifier(**clf.best_params_)

        # Train the new classifier on all training data.
        train_start_time = time.time()
        clf.fit(X_train_new, y_train)
        results[key]['nn_iters'] = clf.n_iter_
        results[key]['train_time'] = round(time.time() - train_start_time, 3)

        # Calculate the final scores.
        results[key]['final_train_score'] = clf.score(X_train_new, y_train)
        results[key]['final_test_score'] = clf.score(X_test_new, y_test)
    except Exception as e:
        print('EXCEPTION!')
        print(e)

pprint.PrettyPrinter(indent=4).pprint(results)
print('total_time:', time.time() - total_start_time)
