import chess


# Class that represents the board states of the puzzle for information
# for the classifier
class PuzzleRepresentaion:

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

    
    def get_all_states(self) -> list:
        """Returns a list of thes fens for each state of the board for all moves in puzzle"""
        states = [self.fen]

        for m in self.moves:

            move = chess.Move.from_uci(m)
            self.board.push(move)
            states.push(self.board.fen())

        return states

    def board(self) -> chess.Board:

        return self.board 

    def theme(self) -> str:

        return self.theme
