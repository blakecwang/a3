#!/usr/bin/env python3

# https://github.com/mGalarnyk/Python_Tutorials/blob/master/Sklearn/PCA/PCA_to_Speed-up_Machine_Learning_Algorithms.ipynb

import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA, FastICA
import matplotlib.pyplot as plt
from scipy.stats import entropy
from numpy.random import randn

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


print('PCA')
print('before', X.shape)
transformer = PCA(n_components=X.shape[1] - 1, random_state=RS)
#transformer = PCA(n_components=0.95, random_state=RS)
X = transformer.fit_transform(X)
print('after', X.shape)
print(transformer.explained_variance_ratio_)
print('sv')
print(transformer.singular_values_)
print('')
exit()

#print('FastICA')
#for i in range(X.shape[1]):
#    hist_stuff(X[:,i], i, 'hist', 'r')

#data = 5 * randn(100) + 50
#x = X[:,9].reshape(-1, 1)
#print(data.shape)
#print(x.shape)
#exit()


#from scipy.stats import anderson
#x = X[:,9]
#result = anderson(x)
#print('Statistic:', result.statistic)
#p = 0
#for i in range(len(result.critical_values)):
#    sl, cv = result.significance_level[i], result.critical_values[i]
#    if result.statistic < result.critical_values[i]:
#        print('data looks normal')
#        print(sl, cv)
#    else:
#        print('data looks NOT normal')
#        print(sl, cv)
#exit()

from scipy.stats import shapiro
for i in range(X.shape[1]):
    x = X[:,i].reshape(-1, 1)
    stat, p = shapiro(x)
    print('shapiro stat', stat)
    print('shapiro p', p)
    print('')
    exit()
#    transformer = FastICA(random_state=RS)
#    x = transformer.fit_transform(x)
#    stat, p = shapiro(x)
#    print('Statistics=%.3f, p=%.3f' % (stat, p))
exit()

from scipy.stats import normaltest
x = X[:,9]
stat, p = normaltest(x)
print(p)
print(stat)
print('')

transformer = FastICA(random_state=RS)
X = transformer.fit_transform(X)

x = X[:,9]
stat, p = normaltest(x)
print(p)
print(stat)

exit()



before = []
after = []
transformer = FastICA(random_state=RS)
n_bins = 50
for i in range(X.shape[1]):
#    hist_stuff(X[:,i], i, 'transformed_hist', 'b')

    x = X[:,i]
    x_min = x.min()
    x_len = x.max() - x_min
    x_bins = [x_min + j * x_len / n_bins for j in range(n_bins)]
    idxs = np.digitize(x, bins=x_bins)
    new_x = [x_bins[idx - 1] for idx in idxs]
    before.append(entropy(new_x, base=2))

X = transformer.fit_transform(X)

for i in range(X.shape[1]):
    x = X[:,i]
    x_min = x.min()
    x_len = x.max() - x_min
    x_bins = [x_min + j * x_len / n_bins for j in range(n_bins)]
    idxs = np.digitize(x, bins=x_bins)
    new_x = [x_bins[idx - 1] for idx in idxs]
    after.append(entropy(new_x, base=2))

b = np.array(before)
a = np.array(after)
print(a)
exit()

print('before', np.mean(b))
print('after',  np.mean(a))
