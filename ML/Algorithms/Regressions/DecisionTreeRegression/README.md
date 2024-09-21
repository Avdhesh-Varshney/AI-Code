# Decision Tree Regression

This module contains an implementation of Decision Tree Regression, a versatile algorithm for predicting a continuous outcome based on input features.

## Usage

To use Decision Tree Regression, follow these steps:

1. Import the `DecisionTreeRegression` class.
2. Create an instance of the class, specifying parameters such as the maximum depth.
3. Fit the model to your training data using the `fit` method.
4. Make predictions using the `predict` method.

Example:

```python
from DecisionTreeRegression import DecisionTreeRegression

dt_model = DecisionTreeRegression(max_depth=3)
dt_model.fit(X_train, y_train)
predictions = dt_model.predict(X_test)
```

## Parameters

- `max_depth`: Maximum depth of the decision tree. Controls the complexity of the model.

## Installation

To use this module, make sure you have the required dependencies installed:

```bash
pip install numpy
```

## Coded By 

[Avdhesh Varshney](https://github.com/Avdhesh-Varshney)

### Happy Coding 👦
