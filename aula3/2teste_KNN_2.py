#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 13:19:06 2018

@author: flaviabernardini
"""

import numpy as np
from sklearn import neighbors, datasets, metrics
from sklearn.neighbors import KNeighborsClassifier

n_neighbors = 15

# import some data to play with
iris = datasets.load_iris()

# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset
iris_X = iris.data
iris_y = iris.target

# print(X)

clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
clf.fit(iris_X, iris_y)

iris_z = clf.predict(iris_X)

print(metrics.classification_report(iris_y, iris_z))

indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test = iris_X[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]

# Create and fit a nearest-neighbor classifier
knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train)
iris_z_test = knn.predict(iris_X_test)
print(metrics.classification_report(iris_y_test, iris_z_test))
