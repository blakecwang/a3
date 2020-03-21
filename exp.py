#!/usr/bin/env python3

import numpy as np

#x = lambda a : a + 10
items = np.array([1,2,3])
new_items = np.array(list(map(lambda x: [x], items)))
print(new_items.shape)

print(new_items[0].__class__)
print(new_items[0])
