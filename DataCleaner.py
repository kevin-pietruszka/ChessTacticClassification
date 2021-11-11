import pandas as pd

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

dataCutter()