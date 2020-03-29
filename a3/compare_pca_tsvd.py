#!/usr/bin/env python3

import numpy as np

def mean_squared_error(list1, list2):
    errors = []
    for i in range(len(list1)):
        errors.append((list1[i] / list2[i] - 1) ** 2)
    return sum(errors) / len(list1)

TSVD_WQ = [1.7e+05, 5.5e+03, 7.9e+02, 2.8e+02, 2.1e+02, 5.3e+01, 1.7e+01, 7.5e+00, 6.7e+00, 5.8e+00, 1.8e+00]
PCA_WQ =  [8.8e+04, 2.6e+03, 7.9e+02, 2.8e+02, 6.2e+01, 4.9e+01, 8.2e+00, 7.5e+00, 6.6e+00, 5.7e+00, 1.2e+00]
TSVD_GB = [233.6]
PCA_GB = [229.7]

print(mean_squared_error(TSVD_WQ, PCA_WQ))
print(mean_squared_error(TSVD_GB, PCA_GB))
