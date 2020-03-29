#!/usr/bin/env python

# docs: https://pymdptoolbox.readthedocs.io/en/latest/index.html
import mdptoolbox, mdptoolbox.example
import sys

stats = False
try:
    if sys.argv[1] == 's':
        stats = True
except Exception:
    exit()

P, R = mdptoolbox.example.forest(
    S=30,
    r1=4,   # wait reward on last state efault 4
    r2=10,  # cut reward on last state, default 2
    p=0.2
)

D = 0.9

vi = mdptoolbox.mdp.ValueIteration(
    transitions=P, # Transition probability matrices.
    reward=R,      # Reward matrices or vectors.
    discount=D     # Discount factor.
)
if not stats:
    # Display the variation of V.
    vi.setVerbose()
vi.run()

if not stats:
    print('split')

pi = mdptoolbox.mdp.PolicyIteration(
    transitions=P,
    reward=R,
    discount=D
)
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
