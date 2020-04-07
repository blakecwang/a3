#!/usr/bin/env python

import sys
import gym
import numpy as np
import mdptoolbox, mdptoolbox.example
import matplotlib.pyplot as plt

np.random.seed(11)

DIR = 'DRUL'
LET = 'HFSG'
S = 30
STATES = S * S
EPSILON = 0.0000001
R_HOLE = -0.75
MAX_STEPS = STATES * 10
N_WALKS = 5

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

D_vi = 0.74
D_pi = 0.74
D_ql = 0.96

print('================ VI,', 'discount =', D_vi)
vi = mdptoolbox.mdp.ValueIteration(P, R, D_vi, epsilon=EPSILON)
vi.setVerbose()
vi.run()
print('vi.time', vi.time)
print('vi.iter', vi.iter)
#vi_times = np.array(vi.times)
#count = vi_times.shape[0]
#vi_times -= np.full(count, vi_times[0])
#vi_results = np.zeros((count, 3))
#last_policy = np.array(vi.policies[count-1])
#for i in range(count):
#    vi_results[i,0] = vi_times[i]
#    rewards = np.zeros(N_WALKS)
#    for j in range(N_WALKS):
#        rewards[j] = walk(P, R, vi.policies[i])
#    vi_results[i,1] = np.mean(rewards)
#    vi_results[i,2] = np.count_nonzero(np.array(vi.policies[i]) != last_policy)

print('================ PI,', 'discount =', D_pi)
pi = mdptoolbox.mdp.PolicyIteration(P, R, D_pi)
pi.run()
print('pi.time', pi.time)
print('pi.iter', pi.iter)
#pi_times = np.array(pi.times)
#count = pi_times.shape[0]
#pi_times -= np.full(count, pi_times[0])
#pi_results = np.zeros((count, 3))
#last_policy = np.array(pi.policies[count-1])
#for i in range(count):
#    pi_results[i,0] = pi_times[i]
#    rewards = np.zeros(N_WALKS)
#    for j in range(N_WALKS):
#        rewards[j] = walk(P, R, pi.policies[i])
#    pi_results[i,1] = np.mean(rewards)
#    pi_results[i,2] = np.count_nonzero(np.array(pi.policies[i]) != last_policy)

print('================ QL,', 'discount =', D_ql)
ql = mdptoolbox.mdp.QLearning(P, R, D_ql, n_iter=1e6)
ql.run()
print('ql.time', ql.time)

#plot_stuff(
#    vi_results[:,0],
#    pi_results[:,0],
#    vi_results[:,1],
#    pi_results[:,1],
#    'VI',
#    'PI',
#    'Time (s)',
#    'Mean Long Term Reward',
#    'converge_lake'
#)
#
#plot_stuff(
#    vi_results[:,0],
#    pi_results[:,0],
#    vi_results[:,2],
#    pi_results[:,2],
#    'VI',
#    'PI',
#    'Time (s)',
#    'Difference from Final Policy',
#    'difference_lake'
#)
