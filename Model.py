import tensorflow as tf
import pandas as pd
import numpy as np
import math

def designModel(num):
    puzzlesdb = np.load("data/matrix_rep%s.npz" % num, allow_pickle=True).values()
    puzzle_labels = np.load("data/label%s.npz" % num, allow_pickle=True).values()
    print("Files Loaded")
    
    for p in puzzlesdb:
        puzzlesdb = p

    for pl in puzzle_labels:
        puzzle_labels = pl

    #Allocates first 80% of database to training
    train_ind = math.floor(len(puzzlesdb) * .8)
    puzzle_train = puzzlesdb[:train_ind]
    train_labels = puzzle_labels[:train_ind]

    #Allocates the rest (20%) to testing
    test_ind = len(puzzlesdb) - train_ind
    puzzle_test = puzzlesdb[:test_ind]
    test_labels = puzzle_labels[:test_ind]
    print("Local Vars Set")

    #Builds the model
    mlp = tf.keras.Sequential()
    mlp.add(tf.keras.layers.Flatten(input_shape=(num+1, 8, 8, 12)))
    mlp.add(tf.keras.layers.Dense(128, activation='relu'))
    mlp.add(tf.keras.layers.Dense(64, activation='relu'))
    #12 Classifications
    mlp.add(tf.keras.layers.Dense(6, activation='softmax'))
    mlp.summary()

    mlp.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
                )

    mlp.fit(puzzle_train, train_labels, epochs=10)
    test_loss, test_acc = mlp.evaluate(puzzle_test, test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)
    print('\nTest loss:', test_loss)

designModel(2)