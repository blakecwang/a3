#!/usr/bin/env python

import sys
import gym
import numpy as np
import mdptoolbox, mdptoolbox.example

stats = False
try:
    if sys.argv[1] == 's':
        stats = True
except Exception:
    print('no args')

LET = 'HFSG'
dim = 5 # must satisfy 4x + 1
np_map = np.zeros((dim, dim), dtype=int)
toggle = True
for i in range(dim):
    if i % 2 == 0:
        np_map[i] = np.ones(dim, dtype=int)
    else:
        idx = dim - 1 if toggle else 0
        np_map[i,idx] = 1
        toggle = not toggle
np_map[0,0] = 2
np_map[dim-1,dim-1] = 3
print(np_map)

custom_map = []
for i in range(dim):
    mystr = ''
    for j in range(dim):
        mystr += LET[np_map[i,j]]
    custom_map.append(mystr)
for line in custom_map:
    print(line)

env = gym.make('FrozenLake-v0', desc=custom_map, is_slippery=True)
#env = gym.make('FrozenLake8x8-v0', is_slippery=True)

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

D = 0.9

vi = mdptoolbox.mdp.ValueIteration(P, R, D)
if not stats:
    # Display the variation of V.
    vi.setVerbose()
vi.run()

if not stats:
    print('split')

pi = mdptoolbox.mdp.PolicyIteration(P, R, D)
if not stats:
    # Display number of different actions between policy n-1 and n
    pi.setVerbose()
pi.run()

if stats:
    print('=======================')
    print('ValueIteration')
#    print('vi.V', vi.V)
    print('vi.policy', vi.policy)
    print('vi.iter', vi.iter)
    print('vi.time', vi.time)
    print('=======================')
    print('PolicyIteration')
#    print('pi.V', pi.V)
    print('pi.policy', pi.policy)
    print('pi.iter', pi.iter)
    print('pi.time', pi.time)
    print('=======================')
    print('same policy:', vi.policy == pi.policy)

DIR = 'ULDR'
print(vi.policy.__class__)
for i, d in enumerate(vi.policy):
    if i % dim == 0:
        print('')
    else:
        print(DIR[d], end='')
print('')
