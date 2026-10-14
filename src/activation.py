"""
Activation functions and analytical first-order derivatives.
"""

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Compute logistic sigmoid non-linearity:
    sigma(x) = 1 / (1 + exp(-x))
    """
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """
    Compute analytical derivative of the sigmoid function:
    sigma'(x) = sigma(x) * (1 - sigma(x))
    """
    s = sigmoid(x)
    return s * (1.0 - s)
