# K Nearest Neighbors Regression

This module contains an implementation of K-Nearest Neighbors Regression, a simple yet effective algorithm for predicting continuous outcomes based on input features.

## Usage

To use K-Nearest Neighbors Regression, follow these steps:

1. Import the `KNNRegression` class.
2. Create an instance of the class, specifying the number of neighbors (`k`).
3. Fit the model to your training data using the `fit` method.
4. Make predictions using the `predict` method.

Example:

```python
from KNearestNeighborsRegression import KNNRegression

knn_model = KNNRegression(k=3)
knn_model.fit(X_train, y_train)
predictions = knn_model.predict(X_test)
```

## Parameters

- `k`: Number of neighbors to consider for prediction.

## Installation

To use this module, make sure you have the required dependencies installed:

```bash
pip install numpy
```

## Coded By 

[Avdhesh Varshney](https://github.com/Avdhesh-Varshney)

### Happy Coding 👦
