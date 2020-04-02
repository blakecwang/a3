#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

filename = 'results.txt'

def plot_stuff(x, y, xlabel, ylabel, name, yticks=None):
    font = { 'family': 'Times New Roman', 'size': 18 }
    plt.rc('font', **font)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.plot(x, y)
    if yticks is not None:
        plt.yticks(yticks)
    plt.tight_layout()
#    plt.savefig(f"{name}.png")
    plt.show()

x_vi = []
y_vi = []
x_pi = []
y_pi = []
results = open(filename, 'r')
before_split = True
for line in results.readlines():
    line = line.strip()
    if line == 'split':
        before_split = False
        continue
    if line[0].isalpha():
        continue

    split_line = line.split()
    if before_split:
        x_vi.append(int(split_line[0]))
        y_vi.append(float(split_line[1]))
    else:
        x_pi.append(int(split_line[0]))
        y_pi.append(int(split_line[1]))
x_vi = np.array(x_vi)
y_vi = np.array(y_vi)
x_pi = np.array(x_pi)
y_pi = np.array(y_pi)

#print('x_vi')
#print(x_vi)
#print('')
#print('y_vi')
#print(y_vi)
#print('')
print('x_pi')
print(x_pi)
print('')
print('y_pi')
print(y_pi)
print('')

#plot_stuff(x_vi, y_vi, 'Iterations', 'Value Variation', 'vi')
plot_stuff(x_pi, y_pi, 'Iterations', 'Policy Actions Changed', 'pi', yticks=list(range(y_pi.max() + 1)))
