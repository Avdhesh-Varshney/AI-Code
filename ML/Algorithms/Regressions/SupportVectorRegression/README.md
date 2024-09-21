# Support Vector Regression (SVR)

This module contains an implementation of Support Vector Regression (SVR), a regression technique using Support Vector Machines (SVM) principles.

## Usage

To use Support Vector Regression, follow these steps:

1. Import the `SupportVectorRegression` class.
2. Create an instance of the class, specifying parameters such as epsilon and C.
3. Fit the model to your training data using the `fit` method.
4. Make predictions using the `predict` method.

Example:

```python
from SupportVectorRegression import SupportVectorRegression

svr_model = SupportVectorRegression(epsilon=0.1, C=1.0)
svr_model.fit(X_train, y_train)
predictions = svr_model.predict(X_test)
```

## Parameters

- `epsilon`: Epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which no penalty is associated in the training loss function.
- `C`: Regularization parameter. The strength of the regularization is inversely proportional to C.

## Installation

To use this module, make sure you have the required dependencies installed:

```bash
pip install numpy
```

## Coded By 

[Avdhesh Varshney](https://github.com/Avdhesh-Varshney)

### Happy Coding 👦
