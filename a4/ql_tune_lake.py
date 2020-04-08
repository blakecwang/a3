#!/usr/bin/env python

import sys
import gym
import numpy as np
import mdptoolbox, mdptoolbox.example
import matplotlib.pyplot as plt

DIR = 'DRUL'
LET = 'HFSG'
S = 30

states = S * S
epsilon = 0.0000001
r_hole = -0.75

n_vals = 20
min_val = 10000
max_val = 1000000
step = (max_val - min_val) / n_vals

n_walks = 100

results = np.zeros((n_vals+1, 4))
#results = np.zeros((n_vals, 2))
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
    max_steps = states * 10
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
        elif r == r_hole:
            print('    Lose!')
            return total_reward
        s = np.random.choice(states, p=P[a,s,:])
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
                R[s,a] = r_hole
            else:
                R[s,a] = reward
        P[a,s,:] /= np.sum(P[a,s,:])

env.close()
mdptoolbox.util.check(P, R)

# all wins at n_iter = 0.7, 0.74, 0.82, 0.84, 0.86
#n_iter = int(1e5)
D = 0.75

for i, n_iter in enumerate(np.arange(min_val, max_val+step, step=step)):
    n_iter = int(n_iter)

#for i in range(n_vals):
#    D = 0.7 + i/100

    print('================ QL-epsilon-greedy n_iter =', n_iter, 'D =', D)
    ql_eps = mdptoolbox.mdp.QLearning(P, R, D, n_iter=n_iter)
    ql_eps.run()
    print('walking...')
    eps_mean_reward = np.mean(np.array([walk(P, R, ql_eps.policy) for _ in range(n_walks)]))

    print('================ QL-random n_iter =', n_iter, 'D =', D)
    ql_rdm = mdptoolbox.mdp.QLearningRandom(P, R, D, n_iter=n_iter)
    ql_rdm.run()
    print('walking...')
    rdm_mean_reward = np.mean(np.array([walk(P, R, ql_rdm.policy) for _ in range(n_walks)]))

    print('================ QL-random n_iter =', n_iter, 'D =', D)
    ql_ucb = mdptoolbox.mdp.QLearningUCB(P, R, D, n_iter=n_iter)
    ql_ucb.run()
    print('walking...')
    ucb_mean_reward = np.mean(np.array([walk(P, R, ql_ucb.policy) for _ in range(n_walks)]))

#    results[i,0] = D
    results[i,0] = n_iter
    results[i,1] = eps_mean_reward
    results[i,2] = rdm_mean_reward
    results[i,3] = ucb_mean_reward

print(results)

plot_stuff(
    results[:,0],
    results[:,1],
    results[:,2],
    results[:,3],
    'Decaying ε-greedy',
    'Random',
    'UCB',
    'Iterations',
    'Mean Long Term Reward',
    'ql_tune_n_iter_lake_strategies'
)

#best_reward = -np.inf
#best_n_iter = None
#for result in results:
#    if result[1] > best_reward:
#        best_reward = result[1]
#        best_n_iter = result[0]
#print('best_reward', best_reward)
#print('best_n_iter', best_n_iter)
