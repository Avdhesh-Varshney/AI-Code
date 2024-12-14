# Linear Regression

This module contains an implementation of the Linear Regression algorithm, a fundamental technique in machine learning for predicting a continuous outcome based on input features.

## Usage

To use the Linear Regression algorithm, follow these steps:

1. Import the `LinearRegression` class.
2. Create an instance of the class.
3. Fit the model to your training data using the `fit` method.
4. Make predictions using the `predict` method.

Example:

```python
from LinearRegression import LinearRegression

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
predictions = lr_model.predict(X_test)
```

## Parameters

- `learning_rate`: The step size for gradient descent.
- `n_iterations`: The number of iterations for gradient descent.

## Installation

To use this module, make sure you have the required dependencies installed:

```bash
pip install numpy
```

## Coder

[Avdhesh Varshney](https://github.com/Avdhesh-Varshney)

### Happy Coding 👦
