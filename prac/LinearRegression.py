#!/usr/bin/env python3

import csv
import itertools as it
import os
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def main():
    # data_path = os.path.join(os.path.dirname(__file__), './LR_wine_data/winequality-red.csv')
    data_path = os.path.join(os.path.dirname(__file__), './LR_wine_data/winequality-white.csv')
    data_file = open(data_path,'r')
    data_reader = csv.reader(data_file, delimiter=';')

    header = data_reader.__next__()
    wine_data = list(map(lambda c: [float(e) for e in c], data_reader))
    data_file.close()

    X = np.array(list(map(lambda c: c[:-1], wine_data)))
    Y = np.array(list(map(lambda c: c[-1], wine_data)))
    # print('header: ', header)
    # LR(X, Y)
    coefs_big = solve_system(X, Y)
    smallerX = PCA(X, n=2)
    coefs_small = solve_system(smallerX, Y)
    print("Regression function with 11 variables:\n\t", pretty_function(coefs_big))
    print("Regression function with 2 variables:\n\t", pretty_function(coefs_small))
    err_big = quality(X, Y, coefs_big)
    err_small = quality(smallerX, Y, coefs_small)
    print(f"Average error for\n\t11 vars: {err_big}\n\tfor 2 vars: {err_small}")
    visualize(smallerX)

def quality(X, Y, a):
    assert X.shape[0] == len(Y)
    assert X.shape[1] == len(a)
    n = len(Y)

    err = 0.0
    for i in range(n):
        err += abs(np.dot(X[i], a) - Y[i])
    
    return err / n

def PCA(X, n=2):
    V, D, U = np.linalg.svd(X, full_matrices=False)
    left = np.matmul(V[:, 0:n], np.diag(D[0:n]))
    res = np.matmul(left, U[0:n, 0:n])

    return res

def visualize(X):
    plt.figure(figsize=(15, 10))
    plt.plot(X[:, 0], X[:, 1], '.', color = '#0000FF', label = 'Xs')
    plt.legend(loc = 'upper right')
    plt.show()

def normalize(X):
    rows, columns = X.shape

    res = np.zeros(shape=(rows, columns))
    tempArray = np.zeros(rows)

    for i in range(columns):
        # Для джижения в центр
        mean = np.mean(X[:, i])
        # Для того, чтобы все было в [0;1]
        std = np.std(X[:, i])
        tempArray = np.empty(0)
        for element in X[:, i]:
            tempArray = np.append(tempArray, ((element - mean) / std))
        res[:, i] = tempArray

    return res

def pretty_function(coefs):
    res = 'y = '
    res += ' + '.join([f'x{i+1} * {coefs[i]:.3f}' for i in range(len(coefs))])
    return res

def LR(X, Y):

    model = LinearRegression()
    model.fit(X, Y)

    return model.coef_
    # testX = np.array([X[42]])
    # testY = Y[42]
    # print('reg function: ', pretty_function(model.coef_))
    # predicted = model.predict(testX)[0]
    # print('predicted score: ', predicted)
    # print('actual score: ', testY)

def solve_system(X, Y):
    res = np.linalg.lstsq(X, Y, rcond=None)
    coefs = res[0]

    return coefs
    # print('Result funtion: ', pretty_function(coefs))
    # testX = np.array(X[42])
    # testY = Y[42]
    # predicted = np.dot(coefs, testX)
    # print('predicted score: ', predicted)
    # print('actual score: ', testY)

if __name__ == "__main__":
    main()
