#!/usr/bin/env python3

import csv
import itertools as it
import os
import numpy as np
from numpy.random import default_rng
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def main():
    X, Y = get_wine_data()
    # X, Y = get_random_data()
    X, _ = normalize(X)
    A, c = RL(X, Y)
    print(pretty_function(c))
    print('Q =', quality(Y, A))

    alphas = []
    qualities = []
    begin = 0.001
    end = 1.0
    steps = 100
    for i in range(steps):
        alpha = begin + (end - begin) / steps * i
        alphas.append(alpha)
        qualities.append(quality(Y, RL(X, Y, alpha = alpha)[0]))
    
    plt.figure(figsize=(15, 10))
    plt.plot(alphas, qualities, color = 'magenta', label = 'Q over alpha')
    plt.legend(loc = 'upper right')
    plt.show()

def get_wine_data():
    # data_path = os.path.join(os.path.dirname(__file__), './LR_wine_data/winequality-red.csv')
    data_path = os.path.join(os.path.dirname(__file__), './LR_wine_data/winequality-white.csv')
    data_file = open(data_path,'r')
    data_reader = csv.reader(data_file, delimiter=';')

    header = data_reader.__next__()
    # print('header: ', header)
    wine_data = list(map(lambda c: [float(e) for e in c], data_reader))
    data_file.close()

    X = np.array(list(map(lambda c: c[:-1], wine_data)))
    Y = np.array(list(map(lambda c: c[-1], wine_data)))
    return X, Y

def get_random_data():
    n_samples = 500
    n_features = 5
    n_redundant = 2
    rng = default_rng()

    Y = rng.random(n_samples)
    X = np.zeros(shape = (n_samples, n_features + n_redundant))
    base = rng.random((n_samples, n_features))
    X[:n_samples, :n_features] = base

    for i in range(n_redundant):
        X[:, n_features + i] = X[:, i] + (-0.05 + rng.random(size = n_samples) * 0.1)

    return X, Y

def quality(A, B):
    n = len(A)

    err = 0.0
    for i in range(n):
        err += pow(A[i] - B[i], 2)
    
    return err

def normalize(X):
    scaler = MinMaxScaler()
    scaler.fit(X)
    return scaler.transform(X), scaler

def pretty_function(coefs):
    res = 'y = '
    res += ' + '.join([f'x{i+1} * {coefs[i]:.3f}' for i in range(len(coefs))])
    return res

def RL(X, Y, alpha = 1.0):
    model = Lasso(alpha = alpha, fit_intercept = False, normalize = False)
    model.fit(X, Y)
    
    return model.predict(X), model.coef_

if __name__ == "__main__":
    main()
