# Huber Regression

This module contains an implementation of Huber Regression, a robust linear regression technique that combines the properties of both least squares and absolute error loss functions. Huber Regression is particularly useful when dealing with datasets that have outliers, as it is less sensitive to outliers compared to standard linear regression.

## Overview

Huber Regression is a regression algorithm that adds a penalty based on the Huber loss function. This loss function is quadratic for small errors and linear for large errors, providing robustness against outliers.

## Usage

To use Huber Regression, follow these steps:

1. Import the `HuberRegression` class.
2. Create an instance of the class, specifying parameters such as the regularization strength (`alpha`), the threshold for the Huber loss function (`epsilon`), the maximum number of iterations (`max_iter`), and the tolerance (`tol`).
3. Fit the model to your training data using the `fit` method.
4. Make predictions using the `predict` method.

Example:

```python
from HuberRegression import HuberRegression

huber_model = HuberRegression(alpha=1.0, epsilon=1.35, max_iter=1000, tol=1e-4)
huber_model.fit(X_train, y_train)
huber_predictions = huber_model.predict(X_test)
```

## Parameters

- `alpha`: The regularization strength. A positive float value.
- `epsilon`: The threshold for the Huber loss function. A positive float value.
- `max_iter`: The maximum number of iterations to run the optimization algorithm.
- `tol`: The tolerance for the optimization. If the updates are smaller than this value, the optimization will stop.

## Installation

To use this module, make sure you have the required dependencies installed:

```bash
pip install numpy
```

## Coded By 

[Utsav Singhal](https://github.com/UTSAVS26)

### Happy Coding 👦