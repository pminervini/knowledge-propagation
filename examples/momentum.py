#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import theano
import theano.tensor as T

from kprop.akp.model import AdaptiveKnowledgePropagation
from kprop.gpkp.model import GaussianProcessKnowledgePropagation

import kprop.visualization.visualization as visualization


def make_loss(Model, l1=0., l2=0.):
    L, y = T.ivector('L'), T.dvector('y')
    mu, eps = T.dscalar('mu'), T.dscalar('eps')
    R, eta = T.dtensor3('R'),  T.dvector('eta')

    loss = Model.loss_symbolic(L, y, mu, R, eta, eps)

    L1 = abs(mu) + T.sum(abs(eta)) + abs(eps)
    L2 = mu ** 2 + T.sum(eta ** 2) + eps ** 2
    regularized_loss = loss + l1 * L1 + l2 * L2

    return theano.function([L, y, mu, R, eta, eps], regularized_loss)


def make_loss_gradient(Model, l1=0., l2=0.):
    L, y = T.ivector('L'), T.dvector('y')
    mu, eps = T.dscalar('mu'), T.dscalar('eps')
    R, eta = T.dtensor3('R'),  T.dvector('eta')

    loss = Model.loss_symbolic(L, y, mu, R, eta, eps)

    L1 = abs(mu) + T.sum(abs(eta)) + abs(eps)
    L2 = mu ** 2 + T.sum(eta ** 2) + eps ** 2
    regularized_loss = loss + l1 * L1 + l2 * L2

    loss_gradient = theano.grad(regularized_loss, [eta, eps])

    return theano.function([L, y, mu, R, eta, eps], loss_gradient)


def make_minimizer(Model):
    L, y = T.ivector('L'), T.dvector('y')
    mu, eps = T.dscalar('mu'), T.dscalar('eps')
    R, eta = T.dtensor3('R'),  T.dvector('eta')

    model = Model(L, y, mu, R, eta, eps)
    return theano.function([L, y, mu, R, eta, eps], model.minimize())


if __name__ == '__main__':
    rows, cols = 8, 8
    N = rows * cols

    edges_vert = [[(i, j), (i, j + 1)] for i in range(rows) for j in range(cols) if i < rows and j < cols - 1]
    edges_horiz = [[(i, j), (i + 1, j)] for i in range(rows) for j in range(cols) if i < rows - 1 and j < cols]

    R = np.zeros((N, N, 2), dtype=theano.config.floatX)

    for [(i, j), (k, l)] in edges_vert:
        row, col = i * rows + j, k * cols + l
        R[row, col, 0] = R[col, row, 0] = 1

    for [(i, j), (k, l)] in edges_horiz:
        row, col = i * rows + j, k * cols + l
        R[row, col, 1] = R[col, row, 1] = 1

    L, y = np.zeros(N, dtype='int8'), np.zeros(N, dtype=theano.config.floatX)

    L[0], y[0] = 1, 1
    L[rows - 1], y[rows - 1] = 1, 1

    L[N - 1], y[N - 1] = 1, -1
    L[N - rows], y[N - rows] = 1, -1

    rs = np.random.RandomState(0)

    mu, eps = 1.0, 1e-4
    eta = rs.normal(loc=0.0, scale=.05, size=(2,))

    #Model = AdaptiveKnowledgePropagation
    Model = GaussianProcessKnowledgePropagation

    f = make_loss(Model, .1, .1)
    gf = make_loss_gradient(Model, .1, .1)

    minimizer_function = make_minimizer(Model)

    prev_gfv = np.array([.0, .0], dtype=theano.config.floatX)
    for i in range(8192):
        hd = visualization.HintonDiagram(is_terminal=True)
        print(hd(minimizer_function(L, y, mu, R, eta, eps).reshape((rows, cols))))

        fv = f(L, y, mu, R, eta, eps)

        print('Loss: %s' % fv)

        gfv = gf(L, y, mu, R, eta, eps)

        # Gradient Descent
        #eta = eta - 1e-3 * gfv[0]

        # Gradient Descent + Momentum
        eta -= 1e-4 * (gfv[0] + 0.5 * prev_gfv)
        prev_gfv = gfv[0]
