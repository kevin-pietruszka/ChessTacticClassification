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




# import numpy as np
# import sklearn
# from sklearn.tree import DecisionTreeClassifier
# import matplotlib.pyplot as plt

# class RandomForest(object):
#     def __init__(self, n_estimators=8, max_depth=3, max_features=0.9):
#         # helper function. You don't have to modify it
#         # Initialization done here
#         self.n_estimators = n_estimators
#         self.max_depth = max_depth
#         self.max_features = max_features
#         self.bootstraps_row_indices = []
#         self.feature_indices = []
#         self.out_of_bag = []
#         self.decision_trees = [sklearn.tree.DecisionTreeClassifier(max_depth=max_depth, criterion='entropy') for i in range(n_estimators)]
        
#     def _bootstrapping(self, num_training, num_features, random_seed = None):
#         """
#         TODO: 
#         - Randomly select a sample dataset of size num_training with replacement from the original dataset. 
#         - Randomly select certain number of features (num_features denotes the total number of features in X, 
#           max_features denotes the percentage of features that are used to fit each decision tree) without replacement from the total number of features.
        
#         Return:
#         - row_idx: the row indices corresponding to the row locations of the selected samples in the original dataset.
#         - col_idx: the column indices corresponding to the column locations of the selected features in the original feature list.
        
#         Reference: https://en.wikipedia.org/wiki/Bootstrapping_(statistics)
        
#         Hint: Please use np.random.choice. First get the row_idx, and then second get the col_idx.
#         """
#         np.random.seed(seed = random_seed) #keep
#         row_idx = np.random.choice(num_training, num_training, replace=True)
#         col_idx = np.random.choice(num_features, int(num_features * self.max_features), replace=False)
#         return row_idx, col_idx
        

            
#     def bootstrapping(self, num_training, num_features):
#         # helper function. You don't have to modify it
#         # Initializing the bootstap datasets for each tree
#         for i in range(self.n_estimators):
#             total = set(list(range(num_training)))
#             row_idx, col_idx = self._bootstrapping(num_training, num_features)
#             total = total - set(row_idx)
#             self.bootstraps_row_indices.append(row_idx)
#             self.feature_indices.append(col_idx)
#             self.out_of_bag.append(total)
            
#     def fit(self, X, y):
#         """
#         TODO:
#         Train decision trees using the bootstrapped datasets.
#         Note that you need to use the row indices and column indices.
        
#         X: NxD numpy array, where N is number 
#            of instances and D is the dimensionality of each 
#            instance
#         y: Nx1 numpy array, the predicted labels

#         Returns: 
#             None. Calling this function should train the decision trees held in self.decision_trees
#         """
#         N, D = X.shape
#         self.bootstrapping(N, D)

#         for i in range(self.n_estimators):

#             rows = self.bootstraps_row_indices[i]
#             cols = self.feature_indices[i]


#             boot = X[rows, :][:, cols]
#             labels = y[rows]

#             tree = self.decision_trees[i]

#             tree.fit(boot, labels)



     
#     def OOB_score(self, X, y):
#         # helper function. You don't have to modify it
#         # This function computes the accuracy of the random forest model predicting y given x.
#         accuracy = []
#         for i in range(len(X)):
#             predictions = []
#             for t in range(self.n_estimators):
#                 if i in self.out_of_bag[t]:
#                     predictions.append(self.decision_trees[t].predict(np.reshape(X[i][self.feature_indices[t]], (1,-1)))[0])
#             if len(predictions) > 0:
#                 accuracy.append(np.sum(predictions == y[i]) / float(len(predictions)))
#         return np.mean(accuracy)

    
#     def plot_feature_importance(self, data_train):
#         """
#         TODO:
#         -Display a bar plot showing the feature importance of every feature in 
#         at least one decision tree from the tuned random_forest from Q3.2.
        
#         Args:
#             data_train: This is the orginal data train Dataframe containg data AND labels.
#                 Hint: you can access labels with data_train.columns
#         Returns:
#             None. Calling this function should simply display the aforementioned feature importance bar chart
#         """
#         tree = self.decision_trees[0] # just going to look at the first tree

#         c = [(0.8 - float(i)/20, 0, 0) for i in range(0, 10)]


#         names = data_train.columns[self.feature_indices[0]]

#         values = tree.feature_importances_
#         plt.rcParams.update({'font.size': 16})
#         plt.figure(figsize=(20, 10))

#         plt.title("Feature importance for Decision Tree")
#         plt.bar(names, values, color = c)
#         plt.show()
