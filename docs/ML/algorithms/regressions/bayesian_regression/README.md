# Bayesian Regression

This module contains an implementation of Bayesian Regression, a probabilistic approach to linear regression that provides uncertainty estimates for predictions.

## Overview

Bayesian Regression is an extension of traditional linear regression that models the distribution of coefficients, allowing for uncertainty in the model parameters. It's particularly useful when dealing with limited data and provides a full probability distribution over the possible values of the regression coefficients.

## Usage

To use Bayesian Regression, follow these steps:

1. Import the `BayesianRegression` class.
2. Create an instance of the class.
3. Fit the model to your training data using the `fit` method.
4. Make predictions using the `predict` method.

Example:

```python
from BayesianRegression import BayesianRegression

blr_model = BayesianRegression()
blr_model.fit(X_train, y_train)
predictions, uncertainties = blr_model.predict(X_test)
```

## Parameters

- `alpha`: Prior precision for the coefficients.
- `beta`: Precision of the noise in the observations.

## Installation

To use this module, make sure you have the required dependencies installed:

```bash
pip install numpy scipy
```

## Coded By 

[Avdhesh Varshney](https://github.com/Avdhesh-Varshney)

### Happy Coding 👦
