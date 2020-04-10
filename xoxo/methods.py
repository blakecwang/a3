#!/usr/bin/env python

import numpy as np
import itertools

# 0 -> ' '
# 1 -> 'O'
# 2 -> 'X'
def print_with_letters(board):
    LETTER_MAP = ' OX'
    for i in range(3):
        letters = [LETTER_MAP[board[i,j]] for j in range(3)]
        print('|' + ' '.join(letters) + '|')

def all_boards():
    return np.array([[i[0:3],i[3:6],i[6:9]] for i in itertools.product(range(3), repeat=9)])

def n_streaks(board, i):
    streak = np.full(3, i)
    n_streaks = 0
    for j in range(3):
        if (board[j,:] == streak).all():
            n_streaks += 1
        if (board[:,j] == streak).all():
            n_streaks += 1
    if (np.diagonal(board) == streak).all():
        n_streaks += 1
    if (np.diagonal(np.rot90(board)) == streak).all():
        n_streaks += 1
    return n_streaks

def is_imbalanced(board):
    o_count = np.count_nonzero(board == 1)
    x_count = np.count_nonzero(board == 2)
    return abs(o_count - x_count) > 1
