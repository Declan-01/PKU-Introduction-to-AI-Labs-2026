# Lab 2 — Machine Learning from Scratch

## Topics

- Logistic regression and stable cross-entropy optimization
- Decision trees and random forests
- Forward/backward propagation with NumPy
- MLP/CNN experiments and hyperparameter tuning

## Public implementation

[`src/ai_labs/ml.py`](../src/ai_labs/ml.py) contains an independent binary
logistic-regression implementation with:

- numerically stable sigmoid evaluation;
- vectorized gradients;
- L2 regularization;
- convergence tracking;
- deterministic tests.

See [`tests/test_ml_nlp.py`](../tests/test_ml_nlp.py).

## Key takeaway

Correct formulas are only the beginning. Shape discipline, numerical stability,
initialization, regularization and reproducible evaluation decide whether a
model trains reliably.
