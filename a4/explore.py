#!/usr/bin/env python

# https://www.slideshare.net/DongMinLee32/exploration-strategies-in-reinforcement-learning-179779846
# https://medium.com/sequential-learning/optimistic-q-learning-b9304d079e11

import math
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return (1 - (1 / math.log(x + 2)))

x = np.arange(10000)
y = np.vectorize(f)(x)
print(y[-1])

plt.plot(x, y)
plt.show()
