# -*- coding: utf-8 -*-

import theano.tensor as T
import theano.tensor.nlinalg as nlinalg

from kprop.gaussianfields.model import GaussianFields

from pylearn2.optimization.batch_gradient_descent import BatchGradientDescent

__author__ = 'pminervini'
__copyright__ = 'INSIGHT Centre for Data Analytics 2016'


class GaussianProcessKnowledgePropagation(GaussianFields):
    def __init__(self, L, y, mu, R, eta, eps):
        """
        Gaussian Process Knowledge Propagation, as described in [1]
        [1] P Minervini et al. - Discovering Similarity and Dissimilarity Relations for Knowledge Propagation
            in Web Ontologies - International Conference on Data Mining (ICDM) 2014

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
    def negative_log_likelihood_symbolic(L, y, mu, R, eta, eps):
        """
        Negative Marginal Log-Likelihood in a Gaussian Process regression model.

        The marginal likelihood  for a set of parameters Theta is defined as follows:

            \log(y|X, \Theta) = - 1/2 y^T K_y^-1 y - 1/2 log |K_y| - n/2 log 2 \pi

        where K_y = K_f + sigma^2_n I is the covariance matrix for the noisy targets y,
        and K_f is the covariance matrix for the noise-free latent f.
        """
        N = L.shape[0]
        W = T.tensordot(R, eta, axes=1)

        large_W = T.zeros((N * 2, N * 2))
        large_W = T.set_subtensor(large_W[:N, :N], 2. * mu * W)
        large_W = T.set_subtensor(large_W[N:, :N], T.diag(L))
        large_W = T.set_subtensor(large_W[:N, N:], T.diag(L))

        large_D = T.diag(T.sum(abs(large_W), axis=0))
        large_M = large_D - large_W

        PrecisionMatrix = T.inc_subtensor(large_M[:N, :N], mu * eps * T.eye(N))

        # Let's try to avoid singular matrices
        _EPSILON = 1e-8
        PrecisionMatrix += _EPSILON * T.eye(N * 2)

        # K matrix in a Gaussian Process regression model
        CovarianceMatrix = nlinalg.matrix_inverse(PrecisionMatrix)

        L_idx = L.nonzero()[0]

        y_l = y[L_idx]
        CovarianceMatrix_L = CovarianceMatrix[N + L_idx, :][:, N + L_idx]

        log_likelihood = 0.

        log_likelihood -= .5 * y_l.T.dot(CovarianceMatrix_L.dot(y_l))
        log_likelihood -= .5 * T.log(nlinalg.det(CovarianceMatrix_L))
        log_likelihood -= .5 * T.log(2 * T.pi)

        return - log_likelihood

    loss_symbolic = negative_log_likelihood_symbolic

    def fit(self, params=None, l1=.0, l2=.0):
        NLL = self.loss_symbolic(self.L, self.y, self.mu, self.R, self.eta, self.eps)

        if params is None:
            params = [self.eta]

        # Symbolic Theano variables that represent the L1 and L2 regularization terms
        L1, L2 = .0, .0
        for param in params:
            L1 += T.sum(abs(param))
            L2 += T.sum(param ** 2)

        regularized_NLL = NLL + l1 * L1 + l2 * L2

        minimizer = BatchGradientDescent(objective=regularized_NLL, params=params, inputs=[], verbose=1)

        minimizer.minimize()
