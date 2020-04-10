#!/usr/bin/env python

import numpy as np
import methods as m
import mdptoolbox, mdptoolbox.example

# 0 -> ' '
# 1 -> 'O'
# 2 -> 'X'

# cells
# |0 1 2|
# |3 4 5|
# |6 7 8|

# Actions: place 'X' on a blank cell
# It doesn't matter who goes first, a policy gives actions for all boards

#m.print_with_letters(boards[1000])

boards = []
rewards = []
for board in m.all_boards():
    if m.is_imbalanced(board):
        continue

    o_streaks = m.n_streaks(board, 1)
    x_streaks = m.n_streaks(board, 2)

    if o_streaks + x_streaks > 1:
        continue

    boards.append(board)
    if x_streaks == 1:
        rewards.append(1)
    elif o_streaks == 1:
        rewards.append(-1)
    else:
        rewards.append(0)

boards = np.array(boards)
rewards = np.array(rewards)

S = boards.shape[0]
A = 9

# P (A × S × S) -> (action x curr_state x next_state)
# R (S × A)     -> (curr_state x action)
P = np.zeros((A, S, S), dtype=np.bool_)
R = np.zeros((S, A), dtype=np.int8)

count = 0
total = S * A
for move in range(A):
    for board_i, board in enumerate(boards):
        next_board = m.next_board(board, move)
        next_i = m.find_index(next_board, boards)
        P[move,board_i,next_i] = 1
        R[board_i,move] = rewards[next_i]

        count += 1
        print(count, '/', total)

np.save('R.npy', R)
np.save('P.npy', P)

#R_load = np.load('R.npy')
#P_load = np.load('P.npy')

mdptoolbox.util.check(P, R)
