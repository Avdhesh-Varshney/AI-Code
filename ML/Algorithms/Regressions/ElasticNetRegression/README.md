# Elastic Net Regression

This module contains an implementation of Elastic Net Regression, a powerful linear regression technique that combines both L1 (Lasso) and L2 (Ridge) regularization. Elastic Net is particularly useful when dealing with high-dimensional datasets and can effectively handle correlated features.

## Usage

To use Elastic Net Regression, follow these steps:

1. Import the ElasticNetRegression class.
2. Create an instance of the class, specifying parameters such as the regularization strength (alpha), the ratio between L1 and L2 regularization (l1_ratio), the maximum number of iterations (max_iter), and the tolerance (tol).
3. Fit the model to your training data using the fit method.
4. Make predictions using the predict method.

## Parameters

- `alpha`: The regularization strength. A positive float value.
- `l1_ratio`: The ratio of L1 regularization to L2 regularization. Should be between 0 and 1.
- `max_iter`: The maximum number of iterations to run the optimization algorithm.
- `tol`: The tolerance for the optimization. If the updates are smaller than this value, the optimization will stop.

## Installation

To use this module, make sure you have the required dependencies installed:

```bash
pip install numpy
```

#### Getting Started

To run the Elastic Net Regression module and the tests, use the following commands:

```bash
python3 ElasticNetRegression.py
python3 test.py
```
#### Output

![alt text](image-1.png)

## Coded By 

[Kamakshi Ojha](https://github.com/KamakshiOjha)

### Happy Coding 👦
