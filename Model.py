import tensorflow as tf
import pandas as pd
import math

puzzlesdb = pd.read_csv("data/puzzle_data.csv", 
                    names=["fen", "moves", "labels"])
puzzle_labels = puzzlesdb.pop("labels")

#Allocates first 80% of database to training
train_ind = math.floor(len(puzzlesdb.index) * .8)
puzzle_train = puzzlesdb.head(train_ind)
train_labels = puzzle_labels.head(train_ind)

#Allocates the rest (20%) to testing
test_ind = len(puzzlesdb.index) - train_ind
puzzle_test = puzzlesdb.tail(test_ind)
test_labels = puzzle_labels.tail(test_ind)

#Builds the model
mlp = tf.keras.Sequential()
mlp.add(tf.keras.Input(2))
mlp.add(tf.keras.layers.Dense(1, activation='relu'))
#12 Classifications
mlp.add(tf.keras.layers.Dense(12, activation='softmax'))
mlp.summary()

mlp.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
            )

mlp.fit(puzzle_train, train_labels, epochs=30)
test_loss, test_acc = mlp.evaluate(puzzle_test,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
print('\nTest loss:', test_loss)