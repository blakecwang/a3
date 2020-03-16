#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

def plot_stuff(x_vals, y_vals, name):
    font = { 'family': 'Arial', 'size': 18 }
    plt.rc('font', **font)
    plt.plot(x_vals, y_vals)
    plt.xlabel('Iterations', fontsize=18, fontname='Arial')
    plt.ylabel('Log-Likelihood Score', fontsize=18, fontname='Arial')
    plt.tight_layout()
#    plt.savefig(f'{name}.png')
    plt.show()

RS = 11

name = 'wq_factor'
tol = 50
target = 'quality'
train = pd.read_csv(f'wine_train.csv')
test = pd.read_csv(f'wine_test.csv')
full = pd.concat([train, test])
y = np.array(train.loc[:, target])
X = np.array(train.drop(target, axis=1))

#name = 'gb_factor'
#tol = 5
#X, y = make_blobs(
#    centers=6,
#    n_features=2,
#    n_samples=1000,
#    random_state=11
#)

transformer = FactorAnalysis(n_components=7, random_state=RS, tol=tol)
X_new = transformer.fit_transform(X)

print(X.shape)
print(X_new.shape)

plot_stuff(range(transformer.n_iter_), transformer.loglike_, name)
