# XGBoost Regressor

This module contains an implementation of the XGBoost Regressor, a popular ensemble learning algorithm that combines the predictions from multiple decision trees to create a more robust and accurate model for regression tasks.

## Usage

To use the XGBoost Regressor, follow these steps:

1. Import the `XGBoostRegressor` class.
2. Create an instance of the class, specifying parameters such as the number of boosting rounds, learning rate, maximum depth, and gamma.
3. Fit the model to your training data using the `fit` method.
4. Make predictions using the `predict` method.

Example:

```python
from XGBoostRegressor import XGBoostRegressor

# Instantiate the XGBoost Regressor
xgb_model = XGBoostRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, gamma=0.1)

# Fit the model to the training data
xgb_model.fit(X_train, y_train)

# Make predictions on the testing data
predictions = xgb_model.predict(X_test)
```

## Parameters

- `n_estimators`: Number of boosting rounds (trees).
- `learning_rate`: Step size shrinkage to prevent overfitting.
- `max_depth`: Maximum depth of each tree.
- `gamma`: Minimum loss reduction required to make a further partition.

## Installation

To use this module, make sure you have the required dependencies installed:

```bash
pip install numpy scikit-learn
```

## Coded By 

[Avdhesh Varshney](https://github.com/Avdhesh-Varshney)

### Happy Coding 👦
