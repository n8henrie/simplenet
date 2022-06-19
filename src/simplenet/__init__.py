"""simplenet :: Simple multilayer perceptron in Python using numpy."""
__version__ = "v0.1.4"
__author__ = "Nathan Henrie"
__email__ = "nate@n8henrie.com"

from simplenet.simplenet import (
    cross_entropy,
    neg_log_likelihood,
    relu,
    sigmoid,
    SimpleNet,
    softmax,
)

__all__ = [
    "cross_entropy",
    "neg_log_likelihood",
    "relu",
    "sigmoid",
    "SimpleNet",
    "softmax",
]
