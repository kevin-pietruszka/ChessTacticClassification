import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Activation, Dropout
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

def CNN(mlp):
    mlp.add(Conv3D(32, (1, 3, 3), activation="relu", input_shape=(num+1, 8, 8, 12), kernel_initializer='he_uniform'))
    mlp.add(MaxPooling3D(2,2,2))
    mlp.add(Dropout(.25))
    mlp.add(Conv3D(32, (1, 3,3), activation="relu", kernel_initializer='he_uniform' ))
    mlp.add(Conv3D(64, (1, 3,3), activation="relu", kernel_initializer='he_uniform' ))
    mlp.add(MaxPooling3D(14,14,14))
    mlp.add(Dropout(.25))
    mlp.add(Flatten())
    mlp.add(Dense(64, activation="relu", kernel_initializer='he_uniform'))
    mlp.add(Dropout(.5))
    #6 Classifications
    mlp.add(tf.keras.layers.Dense(6, activation='softmax',kernel_initializer='he_uniform'))
    return mlp

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
    mlp.add(Dropout(.25))
    mlp.add(tf.keras.layers.Dense(64, activation='relu'))
    mlp.add(Dropout(.25))
    mlp.add(tf.keras.layers.Dense(32, activation='relu'))
    mlp.add(Dropout(.25))
    #12 Classifications
    mlp.add(tf.keras.layers.Dense(18, activation='softmax'))
    mlp.summary()

    opt = tf.keras.optimizers.SGD(learning_rate=.00001)
    mlp.compile(optimizer=opt,
                loss='categorical_crossentropy',
                metrics=['accuracy']
                )

    epochs=20
    history = mlp.fit(puzzle_train, train_labels, epochs= epochs)
    test_loss, test_acc = mlp.evaluate(puzzle_test, test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)
    print('\nTest loss:', test_loss)
    accuracy = history.history['accuracy']
    print(accuracy)
    loss_val = history.history['loss']
    epochs = range(1,epochs+1)
    
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.plot(epochs, accuracy, 'bo', label='accuracy')
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss')
    ax2.plot(epochs, loss_val, 'ro', label='loss')
    ax2.legend(loc="upper right")
    plt.title('Training Accuracy and Loss Over Epochs')
    #Change file name according to optimizer
    plt.savefig("data/plots/SGD%s.png" % num)

designModel(2)
designModel(4)
designModel(6)
designModel(8)
