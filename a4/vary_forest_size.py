#!/usr/bin/env python

# docs: https://pymdptoolbox.readthedocs.io/en/latest/index.html
import mdptoolbox, mdptoolbox.example
import matplotlib.pyplot as plt
import sys
import numpy as np

def plot_stuff(x, y1, y2, y1_label, y2_label, xlabel, ylabel, name):
    plt.clf()
    font = { 'family': 'Times New Roman', 'size': 18 }
    plt.rc('font', **font)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y1, label=y1_label)
    plt.plot(x, y2, label=y2_label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{name}.png")
#    plt.show()

D = 0.95
n_vals = 10
results = np.zeros((n_vals, 6))
for i in range(n_vals):
    S = (i + 1) * 2
#    S = 2 ** (i+1)
    P, R = mdptoolbox.example.forest(S=S)

    vi = mdptoolbox.mdp.ValueIteration(
        transitions=P,
        reward=R,
        discount=D,
        max_iter=1000
    )
#    vi.setVerbose()
    vi.run()

    pi = mdptoolbox.mdp.PolicyIteration(
        transitions=P,
        reward=R,
        discount=D
    )
#    pi.setVerbose()
    pi.run()

    results[i,0] = S
    results[i,1] = vi.iter
    results[i,2] = pi.iter
    results[i,3] = vi.time
    results[i,4] = pi.time
    results[i,5] = vi.policy == pi.policy

plot_stuff(
    np.array(results[:,0], dtype=int),
    results[:,1],
    results[:,2],
    'VI',
    'PI',
    'States',
    'Iterations',
    'iter_compare'
)

plot_stuff(
    np.array(results[:,0], dtype=int),
    results[:,3],
    results[:,4],
    'VI',
    'PI',
    'States',
    'Time (s)',
    'time_compare'
)

print(results[:,5])
