#!/usr/bin/env python3

# https://github.com/mGalarnyk/Python_Tutorials/blob/master/Sklearn/PCA/PCA_to_Speed-up_Machine_Learning_Algorithms.ipynb

import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA, FastICA
import matplotlib.pyplot as plt
from scipy.stats import entropy
from numpy.random import randn
import sys

def hist_stuff(x, i, name, color):
    plt.clf()

    x_min = x.min()
    x_max = x.max()
    x_len = x_max - x_min
    x_range = (x_min - 0.05 * x_len, x_max + 0.05 * x_len)
    plt.hist(x, bins=50, density=True, range=x_range, color=color, ec=color)

    frame1 = plt.gca()
    frame1.axes.xaxis.set_visible(False)
    frame1.axes.yaxis.set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{name}_{i}.png')

RS = 11

try:
    if sys.argv[1] == 'wq':
        print('Using Wine Quality (WQ) dataset...')
        target = 'quality'
        train = pd.read_csv(f'wine_train.csv')
        test = pd.read_csv(f'wine_test.csv')
        full = pd.concat([train, test])
        y = np.array(train.loc[:, target])
        X = np.array(train.drop(target, axis=1))
    elif sys.argv[1] == 'gb':
        print('Using Generated Blobs (GB) dataset...')
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

transformer = PCA(n_components=0.95, random_state=RS)
X_pca = transformer.fit_transform(X)
print('before', X.shape)
print('after PCA', X_pca.shape)
print('explained variance ratio', transformer.explained_variance_ratio_)
print('sv', transformer.singular_values_)
print('')

for i in range(X.shape[1]):
    hist_stuff(X[:,i], i, 'before', 'r')
    hist_stuff(X_pca[:,i], i, 'pca', 'b')

print('FastICA')
transformer = FastICA(random_state=RS)
X_ica = transformer.fit_transform(X)
print('after FastICA', X_ica.shape)

for i in range(X.shape[1]):
    hist_stuff(X_ica[:,i], i, 'ica', 'b')
