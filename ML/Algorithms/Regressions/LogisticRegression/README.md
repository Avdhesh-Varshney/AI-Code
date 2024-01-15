# Logistic Regression

This module contains an implementation of Logistic Regression, a popular algorithm for binary classification.

## Usage

To use Logistic Regression, follow these steps:

1. Import the `LogisticRegression` class.
2. Create an instance of the class, specifying parameters such as learning rate and number of iterations.
3. Fit the model to your training data using the `fit` method.
4. Make predictions using the `predict` method.

Example:

```python
from LogisticRegression import LogisticRegression

logistic_model = LogisticRegression(learning_rate=0.01, n_iterations=1000)
logistic_model.fit(X_train, y_train)
predictions = logistic_model.predict(X_test)
```

## Parameters

- `learning_rate`: Step size for gradient descent.
- `n_iterations`: Number of iterations for gradient descent.

## Installation

To use this module, make sure you have the required dependencies installed:

```bash
pip install numpy
```

## Coded By 

[Avdhesh Varshney](https://github.com/Avdhesh-Varshney)

### Happy Coding 👦
