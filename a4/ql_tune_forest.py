#!/usr/bin/env python

import sys
import gym
import numpy as np
import mdptoolbox, mdptoolbox.example
import matplotlib.pyplot as plt

states = 15

n_vals = 10
min_val = 10000
max_val = 150000
step = (max_val - min_val) / n_vals

n_walks = 1000
n_steps = states * 10

results = np.zeros((n_vals+1, 2))
np.random.seed(11)

def plot_stuff(x, y, xlabel, ylabel, name):
    plt.clf()
    font = { 'family': 'Times New Roman', 'size': 18 }
    plt.rc('font', **font)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y)
    plt.tight_layout()
    plt.savefig(f"{name}.png")
    plt.show()

# P (A × S × S)
# R (S × A)
def walk(P, R, policy):
    total_reward = 0
    s = 0
    i = 0
    while i < n_steps:
        a = policy[s]
        r = R[s,a]
        total_reward += r
        s = np.random.choice(states, p=P[a,s,:])
        i += 1
    return total_reward

P, R = mdptoolbox.example.forest(S=states)
D = 0.99

for i, n_iter in enumerate(np.arange(min_val, max_val+step, step=step)):
    n_iter = int(n_iter)

    print('================ QL n_iter =', n_iter, 'D =', D)
    ql = mdptoolbox.mdp.QLearning2(P, R, D, n_iter=n_iter)
    ql.run()
    print('time', ql.time)
    print(ql.policy)
    print(ql.N)

    print('walking...')
    rewards = np.array([walk(P, R, ql.policy) for _ in range(n_walks)])
    mean_reward = np.mean(rewards)

    results[i,0] = n_iter
    results[i,1] = mean_reward

print(results)

print('================ VI n_iter =', n_iter, 'D =', 0.98)
vi = mdptoolbox.mdp.ValueIteration(P, R, 0.98)
vi.run()
print('walking...')
rewards = np.array([walk(P, R, vi.policy) for _ in range(n_walks)])
mean_reward = np.mean(rewards)
print('VI mean reward', mean_reward)
print(vi.policy)

plot_stuff(
    results[:,0],
    results[:,1],
    'Iterations',
    'Mean Long Term Reward',
    'ql_tune_iter_forest'
)
