import tensorflow as tf
import pandas as pd

puzzlesdb = pd.read_csv("data/puzzle_data.csv", 
                      names=["fen", "moves", "labels"])
puzzle_labels = puzzlesdb.pop("labels")

#Allocates first 80% of database to training
train_ind = len(puzzlesdb.index) * .8
puzzle_train = puzzlesdb.head(train_ind)
train_labels = puzzle_labels.head(train_ind)

#Allocates the rest (20%) to testing
test_ind = len(puzzlesdb.index) - train_ind
puzzle_test = puzzlesdb.tail(test_ind)
test_labels = puzzle_labels.tail(test_ind)

#Builds the model
model = tf.keras.Sequential([
    
])

