#!/usr/bin/env python3

# https://github.com/mGalarnyk/Python_Tutorials/blob/master/Sklearn/PCA/PCA_to_Speed-up_Machine_Learning_Algorithms.ipynb

import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA, FastICA

target = 'quality'
train = pd.read_csv(f'wine_train.csv')
test = pd.read_csv(f'wine_test.csv')
full = pd.concat([train, test])
y = np.array(train.loc[:,target])
X = np.array(train.drop(target, axis=1))

#X, y = make_blobs(
#    centers=6,
#    n_features=2,
#    n_samples=1000,
#    random_state=11
#)

#print(X.shape)
#print(X)
#print('-')

pca = PCA(n_components=0.95)
pca.fit(X)
X = pca.transform(X)

print(X.shape)
print(pca.components_)
print(pca.explained_variance_ratio_)
#print(X)
