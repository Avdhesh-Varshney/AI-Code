# Lasso Regression

This module contains an implementation of Lasso Regression, a linear regression technique with L1 regularization.

## Overview

Lasso Regression is a regression algorithm that adds a penalty term based on the absolute values of the coefficients. This penalty term helps in feature selection by driving some of the coefficients to exactly zero, effectively ignoring certain features.

## Usage

To use Lasso Regression, follow these steps:

1. Import the `LassoRegression` class.
2. Create an instance of the class, specifying parameters like learning rate, lambda (regularization strength), and the number of iterations.
3. Fit the model to your training data using the `fit` method.
4. Make predictions using the `predict` method.

Example:

```python
from LassoRegression import LassoRegression

lasso_model = LassoRegression(learning_rate=0.01, lambda_param=0.1, n_iterations=1000)
lasso_model.fit(X_train, y_train)
predictions = lasso_model.predict(X_test)
```

## Parameters

- `learning_rate`: The step size for gradient descent.
- `lambda_param`: Regularization strength (L1 penalty).
- `n_iterations`: The number of iterations for gradient descent.

## Installation

To use this module, make sure you have the required dependencies installed:

```bash
pip install numpy
```

## Coded By 

[Avdhesh Varshney](https://github.com/Avdhesh-Varshney)

### Happy Coding 👦
