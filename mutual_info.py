#!/usr/bin/env python3

# https://stackoverflow.com/questions/20491028/optimal-way-to-compute-pairwise-mutual-information-using-numpy

import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import mutual_info_score
from sklearn.decomposition import PCA, FastICA

def calc_mutual_info(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

RS = 11

#target = 'quality'
#train = pd.read_csv(f'wine_train.csv')
#test = pd.read_csv(f'wine_test.csv')
#full = pd.concat([train, test])
#y = np.array(train.loc[:, target])
#X = np.array(train.drop(target, axis=1))

X, y = make_blobs(
    centers=6,
    n_features=2,
    n_samples=1000,
    random_state=11
)

np.random.seed(RS)
n = X.shape[1]
mutual_info_matrix = np.zeros((n, n))
bins = 50

mi_sum = 0
count = 0
for ix in np.arange(n):
    for jx in np.arange(ix+1,n):
        mi = calc_mutual_info(X[:,ix], X[:,jx], bins)
        mutual_info_matrix[ix,jx] = mi
        count += 1
        mi_sum += mi
print('before FastICA, mean:', mi_sum / count)
print('================')

best_n_components = None
lowest_mean_mi = np.inf
for n_components in range(2, X.shape[1] + 1):

    transformer = FastICA(random_state=RS, n_components=n_components)
    tf_X = transformer.fit_transform(X)

    mi_sum = 0
    count = 0
    for ix in np.arange(n):
        for jx in np.arange(ix+1,n_components):
            mi = calc_mutual_info(tf_X[:,ix], tf_X[:,jx], bins)
            mutual_info_matrix[ix,jx] = mi
            count += 1
            mi_sum += mi

    mean_mi = mi_sum / count
    print('mean_mi', mean_mi)
    if mean_mi < lowest_mean_mi:
        lowest_mean_mi = mean_mi
        best_n_components = n_components

print('================')
print('lowest_mean_mi', lowest_mean_mi)
print('best_n_components', best_n_components)
