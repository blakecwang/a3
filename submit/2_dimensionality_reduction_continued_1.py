#!/usr/bin/env python3

import pandas as pd
import numpy as np
import sys
from sklearn.random_projection import GaussianRandomProjection
from scipy.spatial import distance
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

def plot_stuff(x_vals, y_vals, name):
    font = { 'family': 'Arial', 'size': 18 }
    plt.rc('font', **font)
    plt.plot(x_vals, y_vals, marker='o')
    plt.xlabel('New Dimension', fontsize=18, fontname='Arial')
    plt.ylabel('Mean Distance Ratio', fontsize=18, fontname='Arial')
    x_min = x_vals.min()
    x_max = x_vals.max()
    x_len = x_max - x_min
    tol = 0.05
    plt.xlim(x_max + tol * x_len, x_min - tol * x_len)
    plt.tight_layout()
    plt.savefig(f'{name}.png')
#    plt.show()

RS = 11

try:
    if sys.argv[1] == 'wq':
        print('Using Wine Quality (WQ) dataset...')
        name = 'wq_random_project'
        target = 'quality'
        train = pd.read_csv(f'wine_train.csv')
        test = pd.read_csv(f'wine_test.csv')
        full = pd.concat([train, test])
        y = np.array(train.loc[:, target])
        X = np.array(train.drop(target, axis=1))
    elif sys.argv[1] == 'gb':
        print('Using Generated Blobs (GB) dataset...')
        name = 'gb_random_project'
        X, y = make_blobs(
            centers=6,
            n_features=2,
            n_samples=1000,
            random_state=11
        )
    else:
        raise Exception
except Exception:
    print("please provide 'wq' or 'gb' as an argument")
    exit(1)


n_pairs = 100
np.random.seed(RS)
sample_idxs = np.random.choice(range(X.shape[0]), size=2*n_pairs, replace=False)
x_vals = np.array(range(1, X.shape[1] + 1))
y_vals = []
for n_components in x_vals:
    print(X.shape[1], '->', n_components)
    np.random.seed(RS)
    best_mean = np.inf
    for rs in np.random.choice(range(1000), size=5, replace=False):
        transformer = GaussianRandomProjection(random_state=rs, n_components=n_components)
        X_new = transformer.fit_transform(X)

        ratios = []
        for i in range(n_pairs):
            idx1 = sample_idxs[2*i]
            idx2 = sample_idxs[2*i+1]
            d_old = distance.euclidean(X[idx1], X[idx2])
            d_new = distance.euclidean(X_new[idx1], X_new[idx2])
            ratio = d_new / d_old
            ratios.append(ratio)
        try_mean = np.mean(ratio)
        if (try_mean - 1)**2 < best_mean:
            best_mean = try_mean
    y_vals.append(best_mean)
    print('best_mean', best_mean)
    print('')

plot_stuff(x_vals, y_vals, name)
