# -*- coding: utf-8 -*-

import numpy as np

import theano
from kprop.gaussianfields.model import GaussianFields

import logging
import unittest


class TestModel(unittest.TestCase):

    def setUp(self):
        pass

    @staticmethod
    def make_minimizer(L, y, mu, W, eps):
        L, y = theano.shared(value=L, name='L'), theano.shared(value=y, name='y')
        mu, eps = theano.shared(value=mu, name='mu'), theano.shared(value=eps, name='eps')
        W = theano.shared(name='W', value=W)
        gf = GaussianFields(L, y, mu, W, eps)
        return theano.function([], gf.minimize())

    def test_model(self):
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

        mu, eps = 1.0, 1e-8

        f = self.make_minimizer(L, y, mu, W, eps)

        self.assertTrue(f()[1] > 0)
        self.assertTrue(f()[N - 2] < 0)


if __name__ == '__main__':
    unittest.main()
