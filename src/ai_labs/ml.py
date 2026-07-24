"""Small NumPy models implemented without a machine-learning framework."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _stable_sigmoid(x: np.ndarray) -> np.ndarray:
    result = np.empty_like(x, dtype=float)
    positive = x >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    result[~positive] = exp_x / (1.0 + exp_x)
    return result


@dataclass
class BinaryLogisticRegression:
    learning_rate: float = 0.1
    l2: float = 0.0
    epochs: int = 1000
    tolerance: float = 1e-8

    def fit(self, x: np.ndarray, y: np.ndarray) -> "BinaryLogisticRegression":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.ndim != 2 or y.shape != (x.shape[0],):
            raise ValueError("expected x.shape=(n, d) and y.shape=(n,)")
        if not np.all(np.isin(y, (0.0, 1.0))):
            raise ValueError("labels must be 0 or 1")

        self.weights_ = np.zeros(x.shape[1], dtype=float)
        self.bias_ = 0.0
        previous_loss = float("inf")

        for epoch in range(self.epochs):
            logits = x @ self.weights_ + self.bias_
            probabilities = _stable_sigmoid(logits)
            error = probabilities - y
            gradient_w = x.T @ error / len(x) + self.l2 * self.weights_
            gradient_b = float(np.mean(error))
            self.weights_ -= self.learning_rate * gradient_w
            self.bias_ -= self.learning_rate * gradient_b

            eps = 1e-12
            data_loss = -np.mean(
                y * np.log(probabilities + eps)
                + (1.0 - y) * np.log(1.0 - probabilities + eps)
            )
            loss = float(data_loss + 0.5 * self.l2 * np.dot(self.weights_, self.weights_))
            if abs(previous_loss - loss) < self.tolerance:
                self.n_iter_ = epoch + 1
                break
            previous_loss = loss
        else:
            self.n_iter_ = self.epochs

        self.loss_ = previous_loss
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "weights_"):
            raise RuntimeError("call fit before prediction")

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted()
        x = np.asarray(x, dtype=float)
        return _stable_sigmoid(x @ self.weights_ + self.bias_)

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(x) >= threshold).astype(int)
