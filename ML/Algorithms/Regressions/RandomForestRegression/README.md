# Random Forest Regression

This module contains an implementation of Random Forest Regression, an ensemble learning method that combines multiple decision trees to create a more robust and accurate model for predicting continuous outcomes based on input features.

## Usage

To use Random Forest Regression, follow these steps:

1. Import the `RandomForestRegression` class.
2. Create an instance of the class, specifying parameters such as the number of trees, maximum depth, and maximum features.
3. Fit the model to your training data using the `fit` method.
4. Make predictions using the `predict` method.

Example:

```python
from RandomForestRegression import RandomForestRegression

rfr_model = RandomForestRegression(n_trees=100, max_depth=5, max_features=2)
rfr_model.fit(X_train, y_train)
predictions = rfr_model.predict(X_test)
```

## Parameters

- `n_trees`: Number of trees in the random forest.
- `max_depth`: Maximum depth of each decision tree.
- `max_features`: Maximum number of features to consider for each split.

## Installation

To use this module, make sure you have the required dependencies installed:

```bash
pip install numpy
```

## Coded By 

[Avdhesh Varshney](https://github.com/Avdhesh-Varshney)

### Happy Coding 👦
