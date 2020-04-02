#!/usr/bin/env python

import sys
import gym
import numpy as np
import mdptoolbox, mdptoolbox.example
import matplotlib.pyplot as plt

DIR = 'DRUL'
LET = 'HFSG'
S = 58
STATES = S * S

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
    np.random.seed(11)
    max_iters = 1000
    total_reward = 0
    s = 0
    i = 0
    while i < max_iters:
        a = policy[s]
        r = R[s,a]
        total_reward += r
        if r != 0:
            break
        s = np.random.choice(STATES, p=P[a,s,:])
        i += 1
    return total_reward

np_map = np.ones((S, S), dtype=int)
toggle = True
hole_len = int(S * 0.3)
offset = S - hole_len
for j in range(S - 2):
    if j % 2 == 0:
        for k in range(hole_len):
            np_map[j,(offset+k)%S] = 0
        toggle = not toggle
        offset += int(S * 0.6)
np_map[0,0] = 2
np_map[S-1,S-1] = 3

custom_map = []
for j in range(S):
    mystr = ''
    for k in range(S):
        mystr += LET[np_map[j,k]]
    custom_map.append(mystr)

env = gym.make('FrozenLake-v0', desc=custom_map, is_slippery=True)
nA, nS = env.nA, env.nS
P = np.zeros([nA, nS, nS])
R = np.zeros([nS, nA])
for s in range(nS):
    for a in range(nA):
        transitions = env.P[s][a]
        for p_trans, next_s, reward, _ in transitions:
            P[a,s,next_s] += p_trans
            if reward == 0:
                R[s,a] = -0.005
            else:
                R[s,a] = reward
        P[a,s,:] /= np.sum(P[a,s,:])

env.close()
mdptoolbox.util.check(P, R)
print('P.shape', P.shape)
print('R.shape', R.shape)

n_vals = 9
results = np.zeros((n_vals, 3))
n_walks = 10
for i in range(n_vals):
    D = (i + 1) / 10

    print('================ VI,', 'discount =', D)
    vi = mdptoolbox.mdp.ValueIteration(P, R, D)
#    vi.setVerbose()
    vi.run()

    print('================ PI,', 'discount =', D)
    pi = mdptoolbox.mdp.PolicyIteration(P, R, D)
#    pi.setVerbose()
    pi.run()

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
    'lake_walk'
)
