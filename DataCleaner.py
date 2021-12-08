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
desiredThemes = ['fork', 'kingsideAttack','sacrifice','pin','discoveredAttack','defensiveMove','advancedPawn',
                 'hangingPiece','deflection','backRankMate','quietMove','attraction','exposedKing','skewer','trappedPiece',
                 'intermezzo','queensideAttack','clearance']


label_dict = {
    'fork'                  : [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'kingsideAttack'        : [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'sacrifice'             : [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'pin'                   : [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'discoveredAttack'      : [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'defensiveMove'         : [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
    'advancedPawn'          : [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
    'hangingPiece'          : [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
    'deflection'            : [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
    'backRankMate'          : [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
    'quietMove'             : [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
    'attraction'            : [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
    'exposedKing'           : [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
    'skewer'                : [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
    'trappedPiece'          : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
    'intermezzo'            : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
    'queensideAttack'       : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
    'clearance'             : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
}


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

# get all puzzle data
def dataCutter():
    data = pd.read_csv('data/lichess_db_puzzle.csv', header=None)
    data = data[[1,2,7]]
    data[7] = data[7].map(lambda x: " ".join([t for t in x.split() if t in desiredThemes]))
    data = data[data[7] != ""]
    
    print(data)
    data.to_csv("data/puzzle_data.csv", index=False, header=False)

# get matrix rep
def dataConverter():
    for file in range(6, 13, 2):
        filename = "data/puzzle_data%s.csv" % file
        data = pd.read_csv(filename, header=None)
        p = []
        for i in range(data.shape[0]):
            if (i % 500 == 0):
                print(i) 
            puzzle = pr(data[0][i], data[1][i], data[2][i].split(" "))
            p.append(puzzle.get_matrix_representation())
        np.savez_compressed("data/matrix_rep%s" % file, np.array(p))

# get fen, moves, themes
def dataSplitter():
    for i in range(2, 13, 2):
        temp = pd.read_csv('data/puzzle_data.csv', header=None)
        temp[1] = temp[1].map(lambda x: " ".join([t for t in x.split() if len(x.split()) == i]))
        print(temp)
        temp = temp[temp[1] != ""]
        print(temp)
        filename = "data/puzzle_data%s.csv" % (i)
        temp.to_csv(filename, index=False, header=False)

# get labels        
def labeler():
    for num in range(6, 13, 2):
        labels = pd.read_csv("data/puzzle_data%s.csv" % num, 
                        names=["fen", "moves", "labels"]).pop("labels")
        labels = labels.map(lambda x: x.split())
        labels = labels.to_numpy()
        all_onehots = np.empty((labels.shape[0], 18))
        i = 0
        for label_list in labels:
            labels_onehot = np.zeros((18))
            for label in label_list:
                labels_onehot += label_dict[label]
            all_onehots[i] = labels_onehot
            i += 1
            
        np.savez_compressed("data/label%s" % num, all_onehots)

# dataCutter()
# print("Cutter complete")
# dataSplitter()
# print("Splitter complete")
dataConverter()
print("Converter complete")
labeler()
print("Labeler complete")