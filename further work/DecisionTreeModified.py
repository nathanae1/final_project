import pandas as pd
import numpy as np
import math
from collections import Counter
from TreeNode import TreeNode


class DecisionTree(object):
    '''
    A decision tree class.
    '''

    def __init__(self, impurity_criterion='entropy', equality='forced'):
        '''
        Initialize an empty DecisionTree.
        '''

        # root Node
        self.root = None
        # string names of features (for interpreting the tree)
        self.feature_names = None
        # Boolean array of whether variable is categorical (or continuous)
        self.categorical = None

        # what type of equality used
        self.equality = self._forced

        # impurity criterion used to make the splits
        if impurity_criterion == 'entropy':
            self.impurity_criterion = self._entropy
        else:
            self.impurity_criterion = self._gini

    def fit(self, X, y, z, feature_names=None):
        '''
        INPUT:
            - X: 2d numpy array
            - y: 1d numpy array
            - z: 1d numpy array
            - feature_names: numpy array of strings
        OUTPUT: None

        Build the decision tree.
        X is a 2 dimensional array with each column being a feature and each
        row a data point.
        y is a 1 dimensional array with each value being the corresponding
        label.
        z is a 1 dimensional array with each value being the corresponding
        removed feature.
        feature_names is an optional list containing the names of each of the
        features.
        '''

        if feature_names is None or len(feature_names) != X.shape[1]:
            self.feature_names = np.arange(X.shape[1])
        else:
            self.feature_names = feature_names

        # Create True/False array of whether the variable is categorical
        self.categorical = np.vectorize(lambda x: isinstance(x, str) or
                                        isinstance(x, bool) or
                                        isinstance(x, unicode))(X[0])

        self.root = self._build_tree(X, y, z)

    def _equality(self):

        return self.impurity_criterion

    def _build_tree(self, X, y, z):
        '''
        INPUT:
            - X: 2d numpy array
            - y: 1d numpy array
            - z: 1d numpy array
        OUTPUT:
            - TreeNode

        Recursively build the decision tree. Return the root node.
        '''

        node = TreeNode()
        index, value, splits = self._choose_split_index(X, y, z)

        if index is None or len(np.unique(y)) == 1:
            node.leaf = True
            node.classes = Counter(y)
            node.name = node.classes.most_common(1)[0][0]
        else:
            X1, y1, X2, y2 = splits
            node.column = index
            node.name = self.feature_names[index]
            node.value = value
            node.categorical = self.categorical[index]
            node.left = self._build_tree(X1, y1)
            node.right = self._build_tree(X2, y2)
        return node

    def _entropy(self, y):
        '''
        INPUT:
            - y: 1d numpy array
        OUTPUT:
            - float

        Return the entropy of the array y.
        '''

        value_counts = np.unique(y, return_counts=True)
        value_counts = value_counts[1] / float(len(y))
        # print value_counts[1]
        return_val = -1 * np.sum(value_counts * np.log2(value_counts))
        return return_val

    def _gini(self, y):
        '''
        INPUT:
            - y: 1d numpy array
        OUTPUT:
            - float

        Return the gini impurity of the array y.
        '''
        value_counts = np.unique(y, return_counts=True)
        value_counts = value_counts[1] / float(len(y))
        return (1 - np.sum(value_counts**2))

    def _make_split(self, X, y, split_index, split_value):
        '''
        INPUT:
            - X: 2d numpy array
            - y: 1d numpy array
            - split_index: int (index of feature)
            - split_value: int/float/bool/str (value of feature)
        OUTPUT:
            - X1: 2d numpy array (feature matrix for subset 1)
            - y1: 1d numpy array (labels for subset 1)
            - X2: 2d numpy array (feature matrix for subset 2)
            - y2: 1d numpy array (labels for subset 2)

        Return the two subsets of the dataset achieved by the given feature and
        value to split on.

        Call the method like this:
        >>> X1, y1, X2, y2 = self._make_split(X, y, split_index, split_value)

        X1, y1 is a subset of the data.
        X2, y2 is the other subset of the data.
        '''
        X1 = X[X[:, split_index] == split_value]
        X2 = X[X[:, split_index] != split_value]
        y1 = y[:len(X1)]
        y2 = y[len(X1):]
        return X1, y1, X2, y2

    def _information_gain(self, y, y1, y2):
        '''
        INPUT:
            - y: 1d numpy array
            - y1: 1d numpy array (labels for subset 1)
            - y2: 1d numpy array (labels for subset 2)
        OUTPUT:
            - float

        Return the information gain of making the given split.

        Use self.impurity_criterion(y) rather than calling _entropy or _gini
        directly.
        '''
        original_y = self.impurity_criterion(y)
        split_y = self.impurity_criterion(y1) * \
            len(y1) / float(len(y)) + \
            self.impurity_criterion(y2) * \
            len(y2) / float(len(y))
        return original_y - split_y

    def _choose_split_index(self, X, y, z):
        '''
        INPUT:
            - X: 2d numpy array
            - y: 1d numpy array
            - z: 1d numpy array
        OUTPUT:
            - index: int (index of feature)
            - value: int/float/bool/str (value of feature)
            - splits: (2d array, 1d array, 2d array, 1d array)

        Determine which feature and value to split on. Return the index and
        value of the optimal split along with the split of the dataset.

        Return None, None, None if there is no split which improves information
        gain.

        Call the method like this:
        >>> index, value, splits = self._choose_split_index(X, y)
        >>> X1, y1, z1, X2, y2, z2 = splits
        '''
        return_index = 0
        return_value = 0.0
        max_gain = 0
        splits = [[], [], [], []]

        X = np.array(X)
        for column in xrange(X.shape[1]):
            if self.categorical[column] is True:
                for unique_value in np.unique(X[:, column]):
                    indiv_gain = \
                        self._information_gain(y,
                                               y[X[:, column] == unique_value],
                                               y[X[:, column] != unique_value],
                                               z[X[:, column] == unique_value],
                                               z[X[:, column] != unique_value])
                    if indiv_gain > max_gain:
                        return_value, return_index, splits = \
                            self.update_values(X, y, column, unique_value)
                        max_gain = indiv_gain
            else:
                for each_value in np.unique(X[:, column]):
                    indiv_gain = \
                        self._information_gain(y,
                                               y[X[:, column] >= each_value],
                                               y[X[:, column] < each_value],
                                               z[X[:, column] >= each_value],
                                               z[X[:, column] < each_value]
                                               )
                    if indiv_gain > max_gain:
                        return_value, return_index, splits = \
                            self.update_values(X, y, column, each_value)
                        max_gain = indiv_gain

        return return_index, return_value, splits

    def update_values(self, X, y, z, column, unique_value):
        return_value = unique_value
        return_index = column
        splits = [X[X[:, column] == unique_value],
                  y[X[:, column] == unique_value],
                  z[X[:, column] == unique_value],
                  X[X[:, column] != unique_value],
                  y[X[:, column] != unique_value],
                  z[X[:, column] != unique_value]
                  ]
        return return_value, return_index, splits

    def predict(self, X):
        '''
        INPUT:
            - X: 2d numpy array
        OUTPUT:
            - y: 1d numpy array

        Return an array of predictions for the feature matrix X.
        '''
        return np.apply_along_axis(self.root.predict_one, axis=1, arr=X)

    def __str__(self):
        '''
        Return string representation of the Decision Tree.
        '''
        return str(self.root)
