# Polynomial Regression

This module contains an implementation of Polynomial Regression, an extension of Linear Regression that models the relationship between the independent variable and the dependent variable as a polynomial.

## Usage

To use Polynomial Regression, follow these steps:

1. Import the `PolynomialRegression` class.
2. Create an instance of the class, specifying the degree of the polynomial.
3. Fit the model to your training data using the `fit` method.
4. Make predictions using the `predict` method.

Example:

```python
from PolynomialRegression import PolynomialRegression

poly_model = PolynomialRegression(degree=2)
poly_model.fit(X_train, y_train)
predictions = poly_model.predict(X_test)
```

## Parameters

- `degree`: Degree of the polynomial.
- `learning_rate`: The step size for gradient descent.
- `n_iterations`: The number of iterations for gradient descent.

## Installation

To use this module, make sure you have the required dependencies installed:

```bash
pip install numpy
```

## Coded By

[Avdhesh Varshney](https://github.com/Avdhesh-Varshney)

### Happy Coding 👦
