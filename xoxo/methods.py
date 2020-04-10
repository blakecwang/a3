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

# Return array of all combinations of 0, 1, 2 on a 3x3 grid.
def all_boards():
    return np.array([[i[0:3],i[3:6],i[6:9]] for i in itertools.product(range(3), repeat=9)])

# Return the number of i-streaks on the given board.
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

# Return True if the differences between 1s and 2s is greater than 1.
def is_imbalanced(board):
    o_count = np.count_nonzero(board == 1)
    x_count = np.count_nonzero(board == 2)
    return abs(o_count - x_count) > 1

# Return the next board given the current board and a move.
def next_board(board, move):
    i = int(move / 3)
    j = move % 3
    new_board = np.copy(board)
    if board[i,j] == 0:
        board[i,j] == 2
    return new_board

# Return the index of board in boards.
def find_index(board, boards):
    i = 0
    while not (boards[i] == board).all():
        i += 1
    return i
