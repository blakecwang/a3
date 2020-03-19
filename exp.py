#!/usr/bin/env python3

import numpy as np
import itertools

def lowest_label_error(labels1, labels2):
    # Get arrays of unique labels.
    unique_labels1 = np.unique(labels1)
    unique_labels2 = np.unique(labels2)

    # Check that the two inputs have the same number of unique labels.
    n_labels = unique_labels1.shape[0]
    if n_labels is not unique_labels2.shape[0]:
        raise Exception('oh no! labels are not the same length!')

    # Init empty masks.
    masks1 = np.zeros((n_labels, labels1.shape[0]), dtype=bool)
    masks2 = np.copy(masks1)

    # Create an array for each unique value and add it to a mask.
    for i in range(n_labels):
        masks1[i] = np.array([label == unique_labels1[i] for label in labels1])
    for i in range(n_labels):
        masks2[i] = np.array([label == unique_labels2[i] for label in labels2])

    print('masks1')
    print(masks1)
    print('masks2')
    print(masks2)

    lowest_error = np.inf
    for masks2_perm in itertools.permutations(masks2):
        masks2_perm = np.array(masks2_perm)
        print('masks2_perm')
        print(masks2_perm)
        error = np.count_nonzero(masks1 != masks2_perm)
        if error < lowest_error:
            lowest_error = error
    return lowest_error / masks1.size

y2 = np.array([1,2,1,2])
y1 = np.array([4,4,4,1])

print(lowest_label_error(y1, y2))
