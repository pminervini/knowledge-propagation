#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import theano

import kprop.gaussianfields.model as model
import kprop.visualization.visualization as visualization

from kprop.linearsystem import InverseSolver, JacobiSolver

def make_minimizer(L, y, mu, W, eps):
    L, y = theano.shared(value=L, name='L'), theano.shared(value=y, name='y')
    mu, eps = theano.shared(value=mu, name='mu'), theano.shared(value=eps, name='eps')
    W = theano.shared(name='W', value=W)

    solver = JacobiSolver(iterations=30)
    gf = model.GaussianFields(L, y, mu, W, eps, solver=solver)
    return theano.function([], gf.minimize())

if __name__ == '__main__':
    rows, cols = 40, 40
    N = rows * cols

    edges = [[(i, j), (i, j + 1)] for i in range(rows) for j in range(cols) if i < rows and j < cols - 1]
    edges += [[(i, j), (i + 1, j)] for i in range(rows) for j in range(cols) if i < rows - 1 and j < cols]

    W = np.zeros((N, N))

    for [(i, j), (k, l)] in edges:
        row, col = i * rows + j, k * cols + l
        W[row, col] = W[col, row] = 1

    L, y = np.zeros(N, dtype='int8'), np.zeros(N)
    L[0], y[0] = 1, 1
    L[rows - 1], y[rows - 1] = 1, 1

    L[N - 1], y[N - 1] = 1, -1
    L[N - rows], y[N - rows] = 1, -1

    for i in range(len(y)):
        if -1e-8 < y[i] < 1e-8:
            y[i] = 0

    mu, eps = 1.0, 1e-8

    minimizer_function = make_minimizer(L, y, mu, W, eps)

    #print(minimizer_function()[:100])

    hd = visualization.HintonDiagram(is_terminal=True)
    print(hd(minimizer_function().reshape((rows, cols))))
