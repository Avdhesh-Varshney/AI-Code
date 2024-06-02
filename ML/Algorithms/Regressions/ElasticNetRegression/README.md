# ElasticNet Regression

This module contains an implementation of the ElasticNet Regression algorithm, which is a linear regression model with combined L1 and L2 regularization.

## Usage

To use the ElasticNet Regression algorithm, follow these steps:

1. Import the `ElasticNetRegression` class.
2. Create an instance of the class.
3. Fit the model to your training data using the `fit` method.
4. Make predictions using the `predict` method.

Example:

```python
from ElasticNetRegression import ElasticNetRegression

# Create an instance of ElasticNetRegression
en_model = ElasticNetRegression()

# Fit the model to the training data
en_model.fit(X_train, y_train)

# Make predictions on the test data
predictions = en_model.predict(X_test)
```

## Parameters

- `learning_rate`: The step size for gradient descent.
- `l1_ratio`: Ratio of L1 regularization in the penalty (0 for Ridge, 1 for Lasso).
- `alpha`: Regularization strength.
- `n_iterations`: The number of iterations for gradient descent.

## Installation

To use this module, make sure you have the required dependencies installed:

```bash
pip install numpy
```

## Coder

[Utsav Singhal](https://github.com/UTSAVS26)

### Happy Coding 👦
