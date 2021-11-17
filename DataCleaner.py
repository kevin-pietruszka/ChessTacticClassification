import pandas as pd
from PuzzleRepresention import PuzzleRepresentation as pr
import numpy as np

identifiedThemes = ['crushing', 'hangingPiece', 'long', 'middlegame', 'advantage', 'endgame', 'short', 
                    'master', 'mate', 'mateIn2', 'fork', 'trappedPiece', 'pin', 'backRankMate', 'masterVsMaster', 
                    'skewer', 'superGM', 'opening', 'discoveredAttack', 'oneMove', 'veryLong', 'exposedKing', 
                    'defensiveMove', 'kingsideAttack', 'rookEndgame', 'advancedPawn', 'deflection', 'promotion',
                    'mateIn1', 'clearance', 'quietMove', 'equality', 'sacrifice', 'knightEndgame', 'pawnEndgame', 
                    'attraction', 'queensideAttack', 'hookMate', 'intermezzo', 'bishopEndgame', 'xRayAttack', 
                    'capturingDefender', 'mateIn3', 'attackingF2F7', 'zugzwang', 'queenEndgame', 'queenRookEndgame', 
                    'interference', 'doubleCheck', 'arabianMate', 'smotheredMate', 'mateIn4', 'anastasiaMate', 
                    'enPassant', 'castling', 'dovetailMate', 'mateIn5', 'doubleBishopMate', 'bodenMate', 'underPromotion']

#Questionable: exposedKing, defensiveMove
desiredThemes = ['hangingPiece', 
                    'fork', 'trappedPiece', 'pin', 'backRankMate',
                    'skewer', 'discoveredAttack', 'exposedKing', 
                    'defensiveMove', 'advancedPawn', 'deflection', 'promotion',
                    'clearance', 'quietMove', 'sacrifice', 
                    'attraction', 'hookMate', 'intermezzo', 'xRayAttack', 
                    'capturingDefender', 'zugzwang', 
                    'interference', 'doubleCheck', 'arabianMate', 'smotheredMate', 'anastasiaMate', 
                    'enPassant', 'castling', 'dovetailMate', 'doubleBishopMate', 'bodenMate', 'underPromotion']

def themeFinder():
    #CSV Format: PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl
    data = pd.read_csv('data/lichess_db_puzzle.csv', header=None)
    #Desired Format: FEN,Moves,Themes(only some)
    data = data[[1,2,7]]
    themes = []
    for ind in data.values:
        words = data[7][ind].split(" ")
        for i in words:
            if i not in themes:
                themes.append(i)
    print(themes)

def dataCutter():
    data = pd.read_csv('data/lichess_db_puzzle.csv', header=None)
    data = data[[1,2,7]]
    data[7] = data[7].map(lambda x: " ".join([t for t in x.split() if t in desiredThemes]))
    data = data[data[7] != ""]
    
    print(data)
    data.to_csv("data/puzzle_data.csv", index=False, header=False)

def dataConverter():
    for file in range(2, 13, 2):
        filename = "data/puzzle_data%s.csv" % file
        data = pd.read_csv(filename, header=None)
        p = []
        for i in range(data.shape[0]):
            if (i % 500 == 0):
                print(i) 
            puzzle = pr(data[0][i], data[1][i], data[2][i].split(" "))
            p.append(puzzle.get_matrix_representation())
        np.savez_compressed("data/matrix_rep%s" % file, *p, dtype=object)

def dataSpliter():
    for i in range(2, 13, 2):
        temp = pd.read_csv('data/puzzle_data.csv', header=None)
        temp[1] = temp[1].map(lambda x: " ".join([t for t in x.split() if len(x.split()) == i]))
        print(temp)
        temp = temp[temp[1] != ""]
        print(temp)
        filename = "data/puzzle_data%s.csv" % (i)
        temp.to_csv(filename, index=False, header=False)


# puzzlesdb = np.load("data/matrix_rep%s.npz" % 2)
# print(len(puzzlesdb))
# for i in range(1, 2):
#     print(puzzlesdb[str(i)])

print("Post data converter")
container = np.load("data/matrix_rep12.npz", allow_pickle=True)
# for v in container.keys():
#     print(v)
print(container.values())
# print(np.shape(container['arr_0']))
# test = [container[key] for key in container]
# print(type(test))
# print(np.shape(test[0]))