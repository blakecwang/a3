#!/usr/bin/env python

import sys
import gym
import numpy as np
import mdptoolbox, mdptoolbox.example
import matplotlib.pyplot as plt

STATES = 15
MAX_STEPS = STATES * 10
EPSILON = 0.01
N_WALKS = 10
np.random.seed(11)

def plot_stuff(x1, x2, y1, y2, y1_label, y2_label, xlabel, ylabel, name):
    plt.clf()
    font = { 'family': 'Times New Roman', 'size': 18 }
    plt.rc('font', **font)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x1, y1, label=y1_label)
    plt.plot(x2, y2, label=y2_label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{name}.png")
#    plt.show()

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

D_vi = 0.98
D_pi = 0.96

print('================ VI,', 'discount =', D_pi)
vi = mdptoolbox.mdp.ValueIteration(P, R, D_vi, epsilon=EPSILON)
vi.run()
vi_times = np.array(vi.times)
count = vi_times.shape[0]
vi_times -= np.full(count, vi_times[0])
vi_results = np.zeros((count, 3))
last_policy = np.array(vi.policies[count-1])
for i in range(count):
    vi_results[i,0] = vi_times[i]
    rewards = np.zeros(N_WALKS)
    for j in range(N_WALKS):
        rewards[j] = walk(P, R, vi.policies[i])
    vi_results[i,1] = np.mean(rewards)
    vi_results[i,2] = np.count_nonzero(np.array(vi.policies[i]) != last_policy)

print('================ PI,', 'discount =', D_pi)
pi = mdptoolbox.mdp.PolicyIteration(P, R, D_pi)
pi.run()
pi_times = np.array(pi.times)
count = pi_times.shape[0]
pi_times -= np.full(count, pi_times[0])
pi_results = np.zeros((count, 3))
last_policy = np.array(pi.policies[count-1])
for i in range(count):
    pi_results[i,0] = pi_times[i]
    rewards = np.zeros(N_WALKS)
    for j in range(N_WALKS):
        rewards[j] = walk(P, R, pi.policies[i])
    pi_results[i,1] = np.mean(rewards)
    pi_results[i,2] = np.count_nonzero(np.array(pi.policies[i]) != last_policy)

plot_stuff(
    vi_results[:,0],
    pi_results[:,0],
    vi_results[:,1],
    pi_results[:,1],
    'VI',
    'PI',
    'Time (s)',
    'Mean Long Term Reward',
    'converge_forest'
)

plot_stuff(
    vi_results[:,0],
    pi_results[:,0],
    vi_results[:,2],
    pi_results[:,2],
    'VI',
    'PI',
    'Time (s)',
    'Difference from Final Policy',
    'difference_forest'
)
