#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import theano
import theano.tensor as T

from kprop.gaussianfields.model import GaussianFields


def make_minimizer():
    L, y = T.ivector('L'), T.dvector('y')
    mu, W, eps = T.dscalar('mu'), T.dmatrix('W'), T.dscalar('eps')
    return theano.function([L, y, mu, W, eps], GaussianFields(L, y, mu, W, eps).minimize())

if __name__ == '__main__':
    # Number of instances
    N = 6

    # Similarity relations between instances (nodes in an undirected graph)
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]

    # Labeled examples
    L = np.array([1, 0, 0, 0, 0, 1], dtype='int8')
    # Labels
    y = np.array([1, 0, 0, 0, 0, -1], dtype='float32')

    # Let's build the similarity matrix: W[i, j] = 1 iff x_i and x_j are similar, 0 otherwise.
    W = np.zeros((N, N))
    for (xi, xj) in edges:
        W[xi, xj] = W[xj, xi] = 1

    # Let's instantiate the function that takes care of the propagation process
    minimizer = make_minimizer()

    print(minimizer(L, y, 1.0, W, 1e-8))
