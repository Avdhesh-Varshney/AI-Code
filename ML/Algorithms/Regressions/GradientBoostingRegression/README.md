# Gradient Boosting Regression

This module contains an implementation of Gradient Boosting Regression, an ensemble learning method that combines multiple weak learners (typically decision trees) to create a more robust and accurate model for predicting continuous outcomes based on input features.

## Usage

To use Gradient Boosting Regression, follow these steps:

1. Import the `GradientBoostingRegression` class.
2. Create an instance of the class, specifying parameters such as the number of estimators, learning rate, and maximum depth.
3. Fit the model to your training data using the `fit` method.
4. Make predictions using the `predict` method.

Example:

```python
from GradientBoostingRegression import GradientBoostingRegression

gbr_model = GradientBoostingRegression(n_estimators=100, learning_rate=0.1, max_depth=3)
gbr_model.fit(X_train, y_train)
predictions = gbr_model.predict(X_test)
```

## Parameters

- `n_estimators`: Number of boosting stages (trees) to be run.
- `learning_rate`: Step size shrinkage to prevent overfitting.
- `max_depth`: Maximum depth of each decision tree.

## Installation

To use this module, make sure you have the required dependencies installed:

```bash
pip install numpy
```

## Coded By 

[Avdhesh Varshney](https://github.com/Avdhesh-Varshney)

### Happy Coding 👦
