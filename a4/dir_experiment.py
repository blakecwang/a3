#!/usr/bin/env python

import sys
import gym
import numpy as np
import mdptoolbox, mdptoolbox.example
import matplotlib.pyplot as plt

D = 0.9
LET = 'HFSG'
# LET = ' FSG'

def policy(custom_map):
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
    vi = mdptoolbox.mdp.ValueIteration(P, R, D)
#    vi.setVerbose()
    vi.run()
    return vi.policy

print('R', policy(['SG','FF'])[0])
print('L', policy(['GS','FF'])[1])
print('U', policy(['GF','SF'])[2])
print('D', policy(['SF','GF'])[0])
