#!/usr/bin/env python

import sys
import gym
import numpy as np
import mdptoolbox, mdptoolbox.example
import matplotlib.pyplot as plt

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
LET = 'HFSG'

n_vals = 10
results = np.zeros((n_vals, 6))
for i in range(n_vals):
    S = 4 * (i + 1) + 1
    np_map = np.zeros((S, S), dtype=int)
    toggle = True
    for j in range(S):
        if j % 2 == 0:
            np_map[j] = np.ones(S, dtype=int)
        else:
            idx = S - 1 if toggle else 0
            np_map[j,idx] = 1
            toggle = not toggle
    np_map[0,0] = 2
    np_map[S-1,S-1] = 3
    print(np_map)

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
                R[s,a] = reward
            P[a,s,:] /= np.sum(P[a,s,:])

    env.close()
    mdptoolbox.util.check(P, R)
    print('P.shape', P.shape)
    print('R.shape', R.shape)

    vi = mdptoolbox.mdp.ValueIteration(P, R, D)
    # vi.setVerbose()
    vi.run()

    pi = mdptoolbox.mdp.PolicyIteration(P, R, D)
    #pi.setVerbose()
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
    'lake_iter_compare'
)

plot_stuff(
    np.array(results[:,0], dtype=int),
    results[:,3],
    results[:,4],
    'VI',
    'PI',
    'States',
    'Time (s)',
    'lake_time_compare'
)

print(results[:,5])
