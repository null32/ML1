#!/usr/bin/env python3

import csv
import itertools as it
import os
import numpy as np
import random as rng
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification

def main():
    # X, Y = random_data()
    X, Y = wine_data()

    ada_classifier = AdaBoostClassifier(n_estimators= 60, random_state= 0)
    ada_classifier.fit(X, Y)

    i = rng.randint(0, len(Y))
    testX = np.array([X[i]])
    testY = Y[i]
    predicted = ada_classifier.predict(testX)[0]

    print(f'Random object from data set:[', ', '.join([f'{x:.3f}' for x in X[i]]), ']')
    print(f'Ada boost predicted: {predicted}, real value: {testY}')

def random_data():
    X, Y = make_classification(
        n_samples= 1000,
        n_features= 5,
        n_informative= 3,
        n_redundant= 0,
        shuffle= False
    )
    return X, Y

def wine_data(red_or_white = True):
    data_path = f'../../prac/LR_wine_data/winequality-{"red" if red_or_white else "white"}.csv'
    data_path = os.path.join(os.path.dirname(__file__), data_path)
    data_file = open(data_path,'r')
    data_reader = csv.reader(data_file, delimiter=';')

    # header
    _ = data_reader.__next__()
    wine_data = list(map(lambda c: [float(e) for e in c], data_reader))
    data_file.close()

    X = np.array(list(map(lambda c: c[:-1], wine_data)))
    Y = np.array(list(map(lambda c: c[-1], wine_data)))
    return X, Y

if __name__ == "__main__":
    main()