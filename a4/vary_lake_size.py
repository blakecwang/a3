#!/usr/bin/env python

import sys
import gym
import numpy as np
import mdptoolbox, mdptoolbox.example
import matplotlib.pyplot as plt
import pprint

def bar_stuff(x, y, xlabel, ylabel, name):
    plt.clf()
    font = { 'family': 'Times New Roman', 'size': 18 }
    plt.rc('font', **font)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    str_x = []
    for num in x:
        str_x.append(str(int(num)))
    plt.xticks(rotation=45)
    plt.bar(str_x,y)
    plt.tight_layout()
    plt.savefig(f"{name}.png")
#    plt.show()

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
#LET = ' FSG'
HOLE_DENSITY = 0.05
np.random.seed(11)

n_vals = 6
results = np.zeros((n_vals, 6))
for i in range(n_vals):
    S = 6 * i + 4
    np_map = np.ones((S, S), dtype=int)
    for _ in range(round(S * S * HOLE_DENSITY)):
        j, k = np.random.choice(S, size=2, replace=True)
        np_map[j,k] = 0
    np_map[0,0] = 2
    np_map[S-1,S-1] = 3

    custom_map = []
    for j in range(S):
        mystr = ''
        for k in range(S):
            mystr += LET[np_map[j,k]]
        custom_map.append(mystr)

#    for line in custom_map:
#        print(line)
#    print('')
#    continue

    env = gym.make('FrozenLake-v0', desc=custom_map, is_slippery=True)
    nA, nS = env.nA, env.nS
    P = np.zeros((nA, nS, nS))
    R = np.zeros((nS, nA))
    for s in range(nS):
        for a in range(nA):
            transitions = env.P[s][a]
            for p_trans, next_s, reward, done in transitions:
                P[a,s,next_s] += p_trans
                if reward == 0.0:
                    if done:
                        R[s,a] = -1
                    else:
                        R[s,a] = 0
                        #R[s,a] = -0.01
            P[a,s,:] /= np.sum(P[a,s,:])

    env.close()
    mdptoolbox.util.check(P, R)
    states = S * S
    print('P.shape', P.shape)
    print('R.shape', R.shape)
    print('R counts', np.unique(R, return_counts=True))

    print('================ VI,', 'states =', states)
    vi = mdptoolbox.mdp.ValueIteration(P, R, D)
    vi.setVerbose()
    vi.run()

    print('================ PI,', 'states =', states)
    pi = mdptoolbox.mdp.PolicyIteration(P, R, D)
    pi.setVerbose()
    pi.run()

    results[i,0] = states
    results[i,1] = vi.iter
    results[i,2] = pi.iter
    results[i,3] = vi.time
    results[i,4] = pi.time
    results[i,5] = np.count_nonzero(np.array(vi.policy) != np.array(pi.policy)) / states

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

bar_stuff(results[:,0], results[:,5], 'States', 'VI-PI Policy Difference', 'lake_policy_diff')

#print(results[:,5])
