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

def plot_stuff(x, y1, y2, y3, y1_label, y2_label, y3_label, xlabel, ylabel, name):
    plt.clf()
    font = { 'family': 'Times New Roman', 'size': 18 }
    plt.rc('font', **font)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y1, label=y1_label)
    plt.plot(x, y2, label=y2_label)
    plt.plot(x, y3, label=y3_label)
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

n_vals = 9
results = np.zeros((n_vals, 4))
n_walks = 100
for i in range(n_vals):
    D = 0.50 + (i + 1) / 50
#for i in range(n_vals):
#    D = 0.90 + (i + 1) / 100
#for i in range(1):
#    D = 0.97

#    print('================ VI,', 'discount =', D)
#    vi = mdptoolbox.mdp.ValueIteration(P, R, D, epsilon=EPSILON)
##    vi.setVerbose()
#    vi.run()
#
#    print('================ PI,', 'discount =', D)
#    pi = mdptoolbox.mdp.PolicyIteration(P, R, D)
##    pi.setVerbose()
#    pi.run()

    print('================ QL,', 'discount =', D)
#    ql = mdptoolbox.mdp.QLearning(P, R, D)
    ql = mdptoolbox.mdp.QLearning(P, R, D, n_iter=int(1e6))
    ql.run()

    print('================ walking,', 'discount =', D)
    vi_rewards = np.zeros(n_walks)
    pi_rewards = np.zeros(n_walks)
    ql_rewards = np.zeros(n_walks)
    for j in range(n_walks):
        vi_rewards[j] = walk(P, R, vi.policy)
        pi_rewards[j] = walk(P, R, pi.policy)
        ql_rewards[j] = walk(P, R, ql.policy)

    vi_mean_reward = np.mean(vi_rewards)
    pi_mean_reward = np.mean(pi_rewards)
    ql_mean_reward = np.mean(ql_rewards)

    results[i,0] = D
    results[i,1] = vi_mean_reward
    results[i,2] = pi_mean_reward
    results[i,3] = ql_mean_reward

plot_stuff(
    results[:,0],
    results[:,1],
    results[:,2],
    results[:,3],
    'VI',
    'PI',
    'QL',
    'Discount',
    'Mean Long Term Reward',
    'forest_walk'
)

best_vi_reward = -np.inf
best_pi_reward = -np.inf
best_ql_reward = -np.inf
best_vi_discount = None
best_pi_discount = None
best_ql_discount = None
for result in results:
    if result[1] > best_vi_reward:
        best_vi_reward = result[1]
        best_vi_discount = result[0]
    if result[2] > best_pi_reward:
        best_pi_reward = result[2]
        best_pi_discount = result[0]
    if result[3] > best_ql_reward:
        best_ql_reward = result[3]
        best_ql_discount = result[0]
print('best_vi_reward', best_vi_reward)
print('best_pi_reward', best_pi_reward)
print('best_ql_reward', best_ql_reward)
print('best_vi_discount', best_vi_discount)
print('best_pi_discount', best_pi_discount)
print('best_ql_discount', best_ql_discount)
print('=======')
print('vi_policy', vi.policy)
print('pi_policy', pi.policy)
print('ql_policy', ql.policy)
print('=======')
print(results)
