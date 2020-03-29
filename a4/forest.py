#!/usr/bin/env python

# docs: https://pymdptoolbox.readthedocs.io/en/latest/index.html
import mdptoolbox, mdptoolbox.example

P, R = mdptoolbox.example.forest(
    S=10,   # The number of states, which should be an integer greater than 1. Default: 3.
    r1=4,   # The reward when the forest is in its last state and action ‘Wait’ is performed. Default: 4.
    r2=50, # The reward when the forest is in its last state and action ‘Cut’ is performed. Default: 2.
    p=0.001,  # The probability of wild fire occurence, in the range ]0, 1[. Default: 0.1.
)

D = 0.9

vi = mdptoolbox.mdp.ValueIteration(
    transitions=P, # Transition probability matrices.
    reward=R,      # Reward matrices or vectors.
    discount=D     # Discount factor.
)
# Display the variation of V.
vi.setVerbose()
vi.run()

print('split')

#print('=======================')
#print('ValueIteration')
#print('vi.V', vi.V)
#print('vi.policy', vi.policy)
#print('vi.iter', vi.iter)
#print('vi.time', vi.time)
#print('=======================')
#print('')

pi = mdptoolbox.mdp.PolicyIteration(
    transitions=P,
    reward=R,
    discount=D
)
# Display number of different actions between policy n-1 and n
pi.setVerbose()
pi.run()

#print('=======================')
#print('PolicyIteration')
#print('pi.V', pi.V)
#print('pi.policy', pi.policy)
#print('pi.iter', pi.iter)
#print('pi.time', pi.time)
#print('=======================')
