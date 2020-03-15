#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.random_projection import GaussianRandomProjection
from scipy.spatial import distance
from sklearn.datasets import make_blobs

RS = 11

target = 'quality'
train = pd.read_csv(f'wine_train.csv')
test = pd.read_csv(f'wine_test.csv')
full = pd.concat([train, test])
y = np.array(train.loc[:, target])
X = np.array(train.drop(target, axis=1))

#X, y = make_blobs(
#    centers=6,
#    n_features=2,
#    n_samples=1000,
#    random_state=11
#)

n_pairs = 10
np.random.seed(RS)
sample_idxs = np.random.choice(range(X.shape[0]), size=2*n_pairs, replace=False)
for n_components in range(1, X.shape[1] + 1):
    transformer = GaussianRandomProjection(random_state=RS, n_components=n_components)
    X_new = transformer.fit_transform(X)
    print(X.shape[1], '->', transformer.n_components_)

    ratios = []
    for i in range(n_pairs):
        idx1 = sample_idxs[2*i]
        idx2 = sample_idxs[2*i+1]
        d_old = distance.euclidean(X[idx1], X[idx2])
        d_new = distance.euclidean(X_new[idx1], X_new[idx2])
        ratio = d_new / d_old
        ratios.append(ratio)
#        print(ratio)
    print('mean', np.mean(ratio))
    print('')

    #print(sample_idxs)
