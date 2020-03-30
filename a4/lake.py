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

DIM = 5 # must satisfy 4x + 1

# +1 for frisbee
# -1 for hole
# 0 for frozen lake

# Left: 0,
# Down: 1,
# Right: 2,
# Up: 3

LET = 'HFSG'
np_map = np.zeros((DIM, DIM), dtype=int)
toggle = True
for i in range(DIM):
    if i % 2 == 0:
        np_map[i] = np.ones(DIM, dtype=int)
    else:
        idx = DIM - 1 if toggle else 0
        np_map[i,idx] = 1
        toggle = not toggle
np_map[0,0] = 2
np_map[DIM-1,DIM-1] = 3
print(np_map)

custom_map = []
for i in range(DIM):
    mystr = ''
    for j in range(DIM):
        mystr += LET[np_map[i,j]]
    custom_map.append(mystr)

#custom_map = [
#    'SGHHH',
#    'HHHHH',
#    'HHHHH',
#    'HHHHH',
#    'HHHHH'
#]

custom_map = [
    'SFFF',
    'FHFH',
    'FFFH',
    'HFFG'
]
DIM=4

#custom_map = [
#    'FF',
#    'GS'
#]
#DIM=2

print('-' * DIM)
for line in custom_map:
    print(line)
print('-' * DIM)

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
# vi.setVerbose()
vi.run()

if not stats:
    print('split')

pi = mdptoolbox.mdp.PolicyIteration(P, R, D)
#pi.setVerbose()
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

#DIR = 'LDRU'
DIR = 'v>^<' # verified using 2x2
for policy in [vi.policy, pi.policy]:
    for i, d in enumerate(policy):
        if i % DIM == 0:
            print('')
        print(DIR[d], end='')
    print('')
