# -*- coding: utf-8 -*-

import theano
import theano.tensor as T

__all__ = [
    "binary_crossentropy",
    "squared_loss",
    "absolute_loss"
]


def binary_crossentropy(predictions, targets):
    """Computes the binary cross-entropy between predictions and targets.

        L(p, t) = - t log(p) - (1 - t) log(1 - p)

    :param predictions: Theano tensor
        Predictions in (0, 1).
    :param targets: Theano tensor
        Targets in [0, 1].
    :return Theano tensor
        An expression for the element-wise binary cross-entropy.
    """
    return T.nnet.binary_crossentropy(predictions, targets)


def squared_loss(predictions, targets):
    """Computes the element-wise squared difference between two tensors.

        L(p, t) = (p - t)^2

    :param predictions: Theano tensor
        Predictions.
    :param targets: Theano tensor
        Targets.
    :return Theano tensor
        An expression for the element-wise squared difference.
    """
    return (predictions - targets) ** 2


def absolute_loss(predictions, targets):
    """Computes the element-wise absolute difference between two tensors.

        L(p, t) = abs(p - t)

    :param predictions: Theano tensor
        Predictions.
    :param targets: Theano tensor
        Targets.
    :return Theano tensor
        An expression for the element-wise absolute difference.
    """
    return T.abs(predictions - targets)
