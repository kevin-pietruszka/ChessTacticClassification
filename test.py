import PuzzleRepresention as PR
import chess
import numpy as np

def main():
    fen = '8/8/2NPkQp1/2R5/1p2n1P1/1B1K1b2/5r2/8 b - - 0 1'
    moves = 'b4b5'
    theme = 'fork'

    puzzle = PR.PuzzleRepresentation(fen, moves, theme)

if __name__ == "__main__":
    main()