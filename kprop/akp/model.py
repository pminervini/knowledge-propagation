# -*- coding: utf-8 -*-

import theano
import theano.tensor as T

from kprop.gaussianfields.model import GaussianFields

import kprop.objectives as objectives
from pylearn2.optimization.batch_gradient_descent import BatchGradientDescent

__author__ = 'pminervini'
__copyright__ = 'INSIGHT Centre for Data Analytics 2016'


class AdaptiveKnowledgePropagation(GaussianFields):
    def __init__(self, L, y, mu, R, eta, eps):
        """
        Adaptive Knowledge Propagation, as described in [1]
        [1] P Minervini et al. - Discovering Similarity and Dissimilarity Relations for Knowledge Propagation
            in Web Ontologies - Journal on Data Semantics, May 2016

        :param L: Theano tensor
            N-length {0, 1} integer vector, where L_i = 1 iff the i-th instance is labeled, and 0 otherwise.
        :param y: Theano tensor
            N-length scalar vector, where y_i is the label of the i-th instance.
        :param mu: Theano tensor
            Scalar regularization parameter.
        :param R: Theano tensor
            NxNxM tensor, where M is the number of similarity relations holding between instances.
        :param eta: Theano tensor
            M-length scalar vector, where mu_i is the weight of the i-th similarity graph.
        :param eps: Theano tensor
            Scalar regularization parameter.
        """
        self.R = R
        self.eta = eta

        W = T.tensordot(self.R, self.eta, axes=1)
        super().__init__(L, y, mu, W, eps)

    @staticmethod
    def leave_one_out_loss_symbolic(L, y, mu, R, eta, eps, loss=objectives.squared_loss):
        """
        The LOO loss for a set of parameters Theta is defined as follows:

            loo(\Theta) = \sum_{i \in L} l(e_{i}^T \hat{f}_{L - {i}}, y_{i}),

        where \hat{f}_{L - {i}} is the result of the propagation process where the i-th example is left out,
        e_{i} is a vector where [e_i]_j = 1 iff i == j, and 0 otherwise, y_{i} is the label of the i-th instance,
        and l(x, y) is a loss function, such as a quadratic loss.
        """
        def leave_one_out_loss(index):
            mask = T.set_subtensor(T.ones_like(L)[index], 0)
            hat_f = AdaptiveKnowledgePropagation(L * mask, y, mu, R, eta, eps).minimize()
            return loss(hat_f[index], y[index])

        loo_losses, _ = theano.scan(lambda index: leave_one_out_loss(index), sequences=[L.nonzero()[0]])
        loo_loss = T.sum(loo_losses)
        return loo_loss

    loss_symbolic = leave_one_out_loss_symbolic

    def fit(self, params=None, l1=.0, l2=.0):
        """
        Fit the model by minimizing the Leave One Out (LOO) loss using gradient-based optimization.
        """
        loo_loss = self.loss_symbolic(self.L, self.y, self.mu, self.R, self.eta, self.eps)

        if params is None:
            params = [self.eta]

        # Symbolic Theano variables that represent the L1 and L2 regularization terms
        L1, L2 = .0, .0
        for param in params:
            L1 += T.sum(abs(param))
            L2 += T.sum(param ** 2)

        regularized_loo_loss = loo_loss + l1 * L1 + l2 * L2

        minimizer = BatchGradientDescent(objective=regularized_loo_loss, params=params, inputs=[], verbose=1)

        minimizer.minimize()
