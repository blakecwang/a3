#!/usr/bin/env python

import sys
import gym
import numpy as np
import mdptoolbox, mdptoolbox.example
import matplotlib.pyplot as plt

STATES = 15
MAX_STEPS = STATES * 10
EPSILON = 0.01
np.random.seed(11)

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
    plt.show()

# P (A × S × S)
# R (S × A)
def walk(P, R, policy):
    total_reward = 0
    s = 0
    i = 0
    while i < MAX_STEPS:
        a = policy[s]
        r = R[s,a]
        total_reward += r
        s = np.random.choice(STATES, p=P[a,s,:])
        i += 1
    return total_reward

P, R = mdptoolbox.example.forest(S=STATES)

n_vals = 99
results = np.zeros((n_vals, 3))
n_walks = 50
for i in range(n_vals):
    D = (i + 1) / 100

    print('================ VI,', 'discount =', D)
    vi = mdptoolbox.mdp.ValueIteration(P, R, D, epsilon=EPSILON)
#    vi.setVerbose()
    vi.run()

    print('================ PI,', 'discount =', D)
    pi = mdptoolbox.mdp.PolicyIteration(P, R, D)
#    pi.setVerbose()
    pi.run()

    print('================ walking,', 'discount =', D)
    vi_rewards = np.zeros(n_walks)
    pi_rewards = np.zeros(n_walks)
    for j in range(n_walks):
        vi_rewards[j] = walk(P, R, vi.policy)
        pi_rewards[j] = walk(P, R, pi.policy)

    vi_mean_reward = np.mean(vi_rewards)
    pi_mean_reward = np.mean(pi_rewards)

    results[i,0] = D
    results[i,1] = vi_mean_reward
    results[i,2] = pi_mean_reward

plot_stuff(
    results[:,0],
    results[:,1],
    results[:,2],
    'VI',
    'PI',
    'Discount',
    'Mean Long Term Reward',
    'forest_walk'
)

best_vi_reward = -np.inf
best_pi_reward = -np.inf
best_vi_discount = None
best_pi_discount = None
for result in results:
    if result[1] > best_vi_reward:
        best_vi_reward = result[1]
        best_vi_discount = result[0]
    if result[2] > best_pi_reward:
        best_pi_reward = result[2]
        best_pi_discount = result[0]
print('best_vi_reward', best_vi_reward)
print('best_pi_reward', best_pi_reward)
print('best_vi_discount', best_vi_discount)
print('best_pi_discount', best_pi_discount)
