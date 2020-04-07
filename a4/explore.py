#!/usr/bin/env python

# https://www.slideshare.net/DongMinLee32/exploration-strategies-in-reinforcement-learning-179779846
# https://medium.com/sequential-learning/optimistic-q-learning-b9304d079e11

import math
import numpy as np
import matplotlib.pyplot as plt

def f(n):
    # probability of greediness decay
#    return 1 - (1 / math.log(n + 1.0001))
    # learning rate decay
    return 1 / math.sqrt(n + 0.00001)

x = np.arange(500000)
y = np.vectorize(f)(x)
print(y[-1])
print(y[100000])

plt.plot(x, y)
plt.show()
