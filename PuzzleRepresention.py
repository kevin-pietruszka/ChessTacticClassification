from typing import final
import chess
import numpy as np
from stockfish import Stockfish
from statistics import mode


piece_dict = {
    'p' : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'n' : [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'b' : [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'r' : [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'q' : [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    'k' : [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    'P' : [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'N' : [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    'B' : [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    'R' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    'Q' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'K' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    '.' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}

# Class that represents the board states of the puzzle for information
# for the classifier
class PuzzleRepresentation:

    def __init__(self, fen, moves, theme) -> None:
        
        # Store the fen representation
        self.fen = fen

        # Uses chess library to create a board rep that can make moves
        # and change the state of the board
        self.board = chess.Board(self.fen)


        # Store the moves of the puzzle as a list
        # Keep track of the current move
        # Store number of moves for potential parameter
        self.moves = moves.split(" ")
        self.next_move = 0
        self.num_moves = len(self.moves)

        # The puzzle theme
        self.theme = theme

        # List of fens
        self.states = self.get_all_states()
        self.matrix_rep = self.get_matrix_representation()
        #self.print_board()

    
    def get_all_states(self) -> list:
        """Returns a list of the fens for each state of the board for all moves in puzzle"""

        states = np.empty((self.num_moves +  1), dtype=object)
        states[0] = self.fen

        for i in range(self.num_moves):
            move = chess.Move.from_uci(self.moves[i])
            self.board.push(move)
            states[i+1] = self.board.fen()

        return states

    def get_matrix_representation(self):
        """Returns a list of each state of the board for all moves in the puzzle in the form of an 8x8x14 matrix"""
        
        mtx = np.full((self.num_moves + 1, 8, 8, 12), piece_dict['.'])

        for i in range(self.num_moves + 1):
            board = chess.Board(self.states[i])
            for square in chess.SQUARES:
                piece = str(board.piece_at(square))
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                if piece != 'None':
                    mtx[i][7 - rank][file] = piece_dict[piece]

        return mtx

    def _material(self):

        init_board = self.states[0].split(' ')[0]
        final_board = self.states[-1].split(' ')[0]

        v=dict(zip('pbnrqPBNRQ',[-1,-3,-3,-5,-9,1,3,3,5,9]))
        
        a = [v.get(c,0) for c in init_board]
        b = [v.get(c,0) for c in final_board]

        return abs(sum(a) - sum(b))


    def tree_features(self):

        sf = Stockfish()

        init = chess.Board(self.states[0])
        final = chess.Board(self.states[-1])

        player_pieces = 'white' if init.turn else 'black'

        sf.set_fen_position(self.states[0])
        tmp = sf.get_evaluation()
        sf.make_moves_from_current_position(self.moves)
        tmp2 = sf.get_evaluation()
        cm = 1 if tmp2['type'] == 'mate' else 0

        p = []
        for i in range(len(self.moves)):
            state = chess.Board(self.states[i])
            pos = self.moves[i][:2]
            piece = state.piece_at(chess.SQUARE_NAMES.index(pos)).symbol()
            p.append(piece)

        return [abs(tmp2['value'] - tmp['value']), cm, mode(p), self._material()]

    def board(self) -> chess.Board:
        return self.board 

    def theme(self) -> str:
        return self.theme
    
    def print_board(self):
        """Prints all board states in puzzle"""
        for board in range(self.matrix_rep.shape[0]):
            for rank in range(8):
                for file in range(8):
                    piece = self.matrix_rep[board][rank][file]
                    for tuple in piece_dict.items():
                        check = True
                        for i in range(12):
                            if piece[i] != tuple[1][i]:
                                check = False
                        if check:
                            print(tuple[0], end=" ")
                print('')
            print('\n')


if __name__ == '__main__':
    # pr = PuzzleRepresentation('r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24', 'f2g3 e6e7 b2b1 b3c1 b1c1 h6c1', 'hangingPiece')
    pr = PuzzleRepresentation('4r1k1/5ppp/r1p5/p1n1RP2/8/2P2N1P/2P3P1/3R2K1 b - - 0 21','e8e5 d1d8 e5e8 d8e8','backRankMate')
    pr._material()
    print(pr.tree_features())