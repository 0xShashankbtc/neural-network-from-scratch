"""
Single-Neuron Feedforward Artificial Neural Network with Backpropagation.
"""

from typing import List, Tuple
import numpy as np
from src.activation import sigmoid, sigmoid_derivative


class SingleNeuronClassifier:
    """
    A single-neuron artificial neural network for binary classification,
    trained via Stochastic Gradient Descent (SGD) with analytical backpropagation.

    Architecture:
        Inputs: x1, x2 (Petal length, Petal width)
        Parameters: w1, w2, b (Weights and scalar bias)
        Linear combination: z = w1*x1 + w2*x2 + b
        Activation: y_hat = sigmoid(z)
        Loss: Squared Error = (y_hat - y)^2
    """

    def __init__(self, random_state: int = 42):
        np.random.seed(random_state)
        self.w1 = float(np.random.randn())
        self.w2 = float(np.random.randn())
        self.b = float(np.random.randn())
        self.costs: List[float] = []

    def forward(self, x1: float, x2: float) -> float:
        """Compute the forward pass for an individual feature pair."""
        z = self.w1 * x1 + self.w2 * x2 + self.b
        return float(sigmoid(z))

    def predict_proba(self, x1: float, x2: float) -> float:
        """Alias for forward pass outputting probability."""
        return self.forward(x1, x2)

    def predict(self, x1: float, x2: float, threshold: float = 0.5) -> int:
        """Classify sample as 0 or 1 based on probability threshold."""
        prob = self.forward(x1, x2)
        return 1 if prob >= threshold else 0

    def fit(
        self,
        dataset: List[List[float]],
        iterations: int = 20000,
        learning_rate: float = 0.1,
        log_interval: int = 100,
    ) -> List[float]:
        """
        Train the network weights using Stochastic Gradient Descent.

        Args:
            dataset: List of rows [length, width, target_class].
            iterations: Number of SGD optimization steps.
            learning_rate: Gradient descent step size (alpha).
            log_interval: Frequency to record dataset-wide loss.

        Returns:
            List of cumulative costs recorded at log intervals.
        """
        self.costs = []
        n_samples = len(dataset)

        for i in range(iterations):
            # Sample random training observation
            idx = np.random.randint(n_samples)
            x1, x2, target = dataset[idx]

            # Forward propagation
            z = self.w1 * x1 + self.w2 * x2 + self.b
            pred = sigmoid(z)

            # Record cumulative loss periodically
            if i % log_interval == 0:
                epoch_loss = 0.0
                for row in dataset:
                    p_z = self.w1 * row[0] + self.w2 * row[1] + self.b
                    p_pred = sigmoid(p_z)
                    epoch_loss += float((p_pred - row[2]) ** 2)
                self.costs.append(epoch_loss)

            # Analytical Backpropagation (Chain Rule)
            # dCost / dPred = 2 * (pred - target)
            dcost_dpred = 2.0 * (pred - target)

            # dPred / dz = sigmoid'(z)
            dpred_dz = sigmoid_derivative(z)

            # Combined error term
            dcost_dz = dcost_dpred * dpred_dz

            # Gradients with respect to parameters
            dcost_dw1 = dcost_dz * x1
            dcost_dw2 = dcost_dz * x2
            dcost_db = dcost_dz * 1.0

            # Parameter update via Gradient Descent
            self.w1 -= learning_rate * dcost_dw1
            self.w2 -= learning_rate * dcost_dw2
            self.b -= learning_rate * dcost_db

        return self.costs
