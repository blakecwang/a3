#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from scipy.stats import entropy
from numpy.random import randn

RS = 11

try:
    if sys.argv[1] == 'wq':
        print('Using Wine Quality (WQ) dataset...')
        name = 'wq_factor'
        target = 'quality'
        train = pd.read_csv(f'wine_train.csv')
        test = pd.read_csv(f'wine_test.csv')
        full = pd.concat([train, test])
        y = np.array(train.loc[:, target])
        X = np.array(train.drop(target, axis=1))
    elif sys.argv[1] == 'gb':
        print('Using Generated Blobs (GB) dataset...')
        name = 'gb_factor'
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


print('before', X.shape)
#transformer = TruncatedSVD(n_components=1, random_state=RS)
transformer = TruncatedSVD(n_components=X.shape[1] - 1, random_state=RS)
X = transformer.fit_transform(X)
print('after', X.shape)
print(transformer.explained_variance_ratio_)
print('sv')
print(transformer.singular_values_)
