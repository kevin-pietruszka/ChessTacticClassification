import math
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np



def run(num):

    num = 10
    puzzlesdb = np.load("data/matrix_tree%s.npz" % num, allow_pickle=True).values()
    puzzle_labels = np.load("data/label_tree%s.npz" % num, allow_pickle=True).values()

    
    for p in puzzlesdb:
        puzzlesdb = p

    for pl in puzzle_labels:
        puzzle_labels = pl


    for p in puzzlesdb:

        p[2] = float(ord(p[2]))

    #Allocates first 80% of database to training
    train_ind = math.floor(len(puzzlesdb) * .8)
    puzzle_train = puzzlesdb[:train_ind, :]
    train_labels = puzzle_labels[:train_ind, :]


    #Allocates the rest (20%) to testing
    puzzle_test = puzzlesdb[train_ind:, :]
    test_labels = puzzle_labels[train_ind:, :]

    tree = sklearn.tree.DecisionTreeClassifier(max_depth=8, criterion='entropy')

    tree.fit(puzzle_train, train_labels)

    puzzle_predicted = []
    for puzzle in puzzle_test:

        puzzle_predicted.append(tree.predict(np.reshape(puzzle, (1, -1))))

    _sum = 0

    puzzle_predicted = np.array(puzzle_predicted)
    test_labels = np.array(test_labels)

    print(puzzle_predicted[0][0])
    print(test_labels[0])

    for i in range(len(puzzle_predicted)):

        pred = puzzle_predicted[i][0]
        act = test_labels[i]

        cmp = pred == act
        if cmp.all():
            _sum += 1

    print(float(_sum) / len(puzzle_predicted))


if __name__ == "__main__":

    run(2) 
    run(4)
    run(6)
    run(8)


