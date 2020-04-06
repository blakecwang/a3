#!/usr/bin/env python

import sys
import gym
import numpy as np
import mdptoolbox, mdptoolbox.example
import matplotlib.pyplot as plt

DIR = 'DRUL'
LET = 'HFSG'
S = 30
STATES = S * S
EPSILON = 0.0000001
R_HOLE = -0.75
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
    #print(np.unique(R, return_counts=True))
    max_steps = STATES * 10
    total_reward = 0
    s = 0
    i = 0
    while i < max_steps:
        a = policy[s]
        r = R[s,a]
        total_reward += r
        if r == 1:
            print('Win!', i, 'steps')
            return total_reward
        elif r == R_HOLE:
            print('    Lose!')
            return total_reward
        s = np.random.choice(STATES, p=P[a,s,:])
        i += 1
    print('        Max!')
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
        for p_trans, next_s, reward, done in transitions:
            P[a,s,next_s] += p_trans
            if done and reward == 0:
                R[s,a] = R_HOLE
            else:
                R[s,a] = reward
        P[a,s,:] /= np.sum(P[a,s,:])

env.close()
mdptoolbox.util.check(P, R)
print('P.shape', P.shape)
print('R.shape', R.shape)

#n_vals = 99
n_vals = 9

results = np.zeros((n_vals, 4))
n_walks = 10
for i in range(n_vals):

#    D = (i + 1) / 100
    D = (i + 1) / 10

    print('================ VI,', 'discount =', D)
    vi = mdptoolbox.mdp.ValueIteration(P, R, D, epsilon=EPSILON)
#    vi.setVerbose()
    vi.run()

    print('================ PI,', 'discount =', D)
    pi = mdptoolbox.mdp.PolicyIteration(P, R, D)
#    pi.setVerbose()
    pi.run()

    print('================ QL,', 'discount =', D)
    ql = mdptoolbox.mdp.QLearning(P, R, D)
#    ql.setVerbose()
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
    'lake_walk'
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
        best_ql_reward = result[2]
        best_ql_discount = result[0]
print('best_vi_reward', best_vi_reward)
print('best_pi_reward', best_pi_reward)
print('best_ql_reward', best_ql_reward)
print('best_vi_discount', best_vi_discount)
print('best_pi_discount', best_pi_discount)
print('best_ql_discount', best_ql_discount)
