#!/usr/bin/env python3

# https://github.com/mGalarnyk/Python_Tutorials/blob/master/Sklearn/PCA/PCA_to_Speed-up_Machine_Learning_Algorithms.ipynb

import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA, FastICA

RS = 11

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


#print('PCA')
#print('before', X.shape)
#transformer = PCA(n_components=0.95, random_state=RS)
#X = transformer.fit_transform(X)
#print('after', X.shape)
#print(transformer.explained_variance_ratio_)
#print('')

print('FastICA')
transformer = FastICA(random_state=RS)
X = transformer.fit_transform(X)
