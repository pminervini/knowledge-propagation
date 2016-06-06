# -*- coding: utf-8 -*-

import abc

import theano
import theano.tensor as T
import theano.tensor.nlinalg as nlinalg
import theano.tensor.slinalg as slinalg

__author__ = 'pminervini'
__copyright__ = 'INSIGHT Centre for Data Analytics 2016'


class ASolver:
    @abc.abstractmethod
    def __call__(self, A, b, inference=False):
        while False:
            yield None


class InverseSolver(ASolver):
    def __call__(self, A, b, inference=False):
        if inference is True:
            solve = slinalg.Solve()
            x = solve(A, b)
        else:
            x = nlinalg.matrix_inverse(A).dot(b)
        return x


class JacobiSolver(ASolver):
    def __init__(self, iterations=0):
        self.iterations = iterations

    def __call__(self, A, b, inference=False):
        dA = T.diagonal(A)
        D = T.diag(dA)
        R = A - D

        iD = T.diag(1.0 / dA)

        x = T.zeros_like(b)
        for i in range(self.iterations):
            x = iD.dot(b - R.dot(x))

        return x
