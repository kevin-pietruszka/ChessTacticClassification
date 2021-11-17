import tensorflow as tf
import pandas as pd
import numpy as np
import math

def designModel(num):
    # puzzlesdb = pd.read_csv("data/puzzle_data%s.csv" % num, 
    #                     names=["fen", "moves", "labels"]).pop("labels")
    puzzlesdb_view = np.load("data/matrix_rep%s.npz" % num, allow_pickle=True)
    puzzle_labels = pd.read_csv("data/puzzle_data%s.csv" % num, 
                        names=["fen", "moves", "labels"]).pop("labels")
    print("Files Loaded")
    
    puzzlesdb = np.empty((len(puzzlesdb_view.values()), num+1, 8, 8, 12))
    i = 0
    for puz in puzzlesdb_view.values():
        puzzlesdb[i] = puz
        i += 1

    #Allocates first 80% of database to training
    train_ind = math.floor(len(puzzlesdb) * .8)
    puzzle_train = puzzlesdb[:train_ind]
    train_labels = puzzle_labels.head(train_ind)

    #Allocates the rest (20%) to testing
    test_ind = len(puzzlesdb) - train_ind
    puzzle_test = puzzlesdb[:test_ind]
    test_labels = puzzle_labels.tail(test_ind)
    print("Local Vars Set")

    #Builds the model
    mlp = tf.keras.Sequential()
    mlp.add(tf.keras.layers.Flatten(input_shape=(num+1, 8, 8, 12)))
    mlp.add(tf.keras.layers.Dense(128, activation='relu'))
    mlp.add(tf.keras.layers.Dense(64, activation='relu'))
    mlp.add(tf.keras.layers.Dense(32, activation='relu'))
    #12 Classifications
    mlp.add(tf.keras.layers.Dense(12, activation='softmax'))
    mlp.summary()

    mlp.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
                )


    mlp.fit(puzzle_train, train_labels, epochs=10)
    test_loss, test_acc = mlp.evaluate(puzzle_test,  test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)
    print('\nTest loss:', test_loss)

designModel(12)