#!/usr/bin/env python

import numpy as np
import methods as m

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
print(boards.shape)
print(rewards.shape)
print(boards)
