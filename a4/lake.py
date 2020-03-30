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

# 0	Move Left
# 1	Move Down
# 2	Move Right
# 3	Move Up

custom_map = [
    'SFFHF',
    'HFHFF',
    'HFFFH',
    'HHHFH',
    'HFFFG'
]
env = gym.make('FrozenLake-v0', desc=custom_map)
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

