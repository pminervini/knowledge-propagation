#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import theano

from kprop.akp.model import AdaptiveKnowledgePropagation
from kprop.gpkp.model import GaussianProcessKnowledgePropagation

import kprop.visualization.visualization as visualization

import logging


def make_model(L, y, mu, R, eta, eps):
    L, y = theano.shared(value=L, name='L'), theano.shared(value=y, name='y')
    mu, eps = theano.shared(value=mu, name='mu'), theano.shared(value=eps, name='eps')
    R, eta = theano.shared(name='R', value=R), theano.shared(name='eta', value=eta)

    #akp = AdaptiveKnowledgePropagation(L, y, mu, R, eta, eps)
    akp = GaussianProcessKnowledgePropagation(L, y, mu, R, eta, eps)

    return akp

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    rows, cols = 10, 10
    N = rows * cols

    edges_vert = [[(i, j), (i, j + 1)] for i in range(rows) for j in range(cols) if i < rows and j < cols - 1]
    edges_horiz = [[(i, j), (i + 1, j)] for i in range(rows) for j in range(cols) if i < rows - 1 and j < cols]

    R = np.zeros((N, N, 2), dtype=theano.config.floatX)

    # Vertical edges connect entities with dissimilar labels
    for [(i, j), (k, l)] in edges_vert:
        row, col = i * rows + j, k * cols + l
        R[row, col, 0] = R[col, row, 0] = 1

    # Horizontal edges connect entities with similar labels
    for [(i, j), (k, l)] in edges_horiz:
        row, col = i * rows + j, k * cols + l
        R[row, col, 1] = R[col, row, 1] = 1

    L, y = np.zeros(N, dtype='int8'), np.zeros(N)
    L[0], y[0] = 1, 1
    L[rows - 1], y[rows - 1] = 1, 1

    L[N - 1], y[N - 1] = 1, -1
    L[N - rows], y[N - rows] = 1, -1

    rs = np.random.RandomState(0)

    mu, eps = 1.0, 1e-4
    eta = rs.normal(loc=0.0, scale=.05, size=(2,))

    akp = make_model(L, y, mu, R, eta, eps)

    minimizer = theano.function([], akp.minimize())
    logging.info('Initial output of the model, before any training process.')
    hd = visualization.HintonDiagram(is_terminal=True)
    print(hd(minimizer().reshape((rows, cols))))

    akp.fit()

    minimizer = theano.function([], akp.minimize())
    logging.info('If the learning process worked, you should see a green and red grid.')
    hd = visualization.HintonDiagram(is_terminal=True)
    print(hd(minimizer().reshape((rows, cols))))