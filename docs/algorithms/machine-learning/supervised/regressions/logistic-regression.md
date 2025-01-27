# 🧮 Logistic Regression Algorithm

<div align="center"> <img src="https://static.javatpoint.com/tutorial/machine-learning/images/logistic-regression-in-machine-learning.png" alt="Logistic Regression Poster" /> </div>

## 🎯 Objective 
Logistic Regression is a supervised learning algorithm used for classification tasks. It predicts the probability of a data point belonging to a particular class, mapping the input to a value between 0 and 1 using a logistic (sigmoid) function.

## 📚 Prerequisites 
- Basic understanding of Linear Algebra and Probability.
- Familiarity with the concept of classification.
- Libraries: NumPy, Pandas, Matplotlib, Scikit-learn.

--- 

## 🧩 Inputs 
- *Input Dataset*: A structured dataset with features (independent variables) and corresponding labels (dependent variable).
- The dependent variable should be categorical (binary or multiclass).
- Example: A CSV file with columns like `age`, `income`, and `purchased` (label).


## 📤 Outputs 
- *Predicted Class*: The output is the probability of each data point belonging to a class.
- *Binary Classification*: Outputs 0 or 1 (e.g., Yes or No).
- *Multiclass Classification*: Outputs probabilities for multiple categories.

---

## 🏛️ Algorithm Architecture 

### 1. Hypothesis Function
The hypothesis function of Logistic Regression applies the sigmoid function:

\[
h_\theta(x) = \frac{1}{1 + e^{-\theta^T x}}
\]

---

### 2. Cost Function
The cost function used in Logistic Regression is the log-loss (or binary cross-entropy):

\[
J(\theta) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]
\]

---

### 3. Gradient Descent
The parameters of the logistic regression model are updated using the gradient descent algorithm:

\[
\theta := \theta - \alpha \frac{\partial J(\theta)}{\partial \theta}
\]

---

## 🏋️‍♂️ Training Process 
- **Model**: Logistic Regression model from sklearn.

- **Validation Strategy**: A separate portion of the dataset can be reserved for validation (e.g., 20%), but this is not explicitly implemented in the current code.

- **Training Data**: The model is trained on the entire provided dataset.



---

## 📊 Evaluation Metrics 
- Accuracy is used to evaluate the classification performance of the model.

\[
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
\]

Where:

- **TP**: True Positives
- **TN**: True Negatives
- **FP**: False Positives
- **FN**: False Negatives

---

## 💻 Code Implementation 

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Generate Example Dataset
np.random.seed(42)
X = np.random.rand(100, 2)  # Features
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Labels: 0 or 1 based on sum of features

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)

# Output Accuracy
print("Accuracy:", accuracy)
```

## 🔍 Scratch Code Explanation
1. **Dataset Generation**:

    - A random dataset with 100 samples and 2 features is created.

    - Labels (`y`) are binary, determined by whether the sum of feature values is greater than 1.

2. **Model Training**:
    - The `LogisticRegression` model from `sklearn` is initialized and trained on the dataset using the fit method.

3. **Predictions**:

    - The model predicts the labels for the input data (`X`) using the `predict` method.

    - The `accuracy_score` function evaluates the accuracy of the predictions.

4. **Output**:

    - The calculated accuracy is printed to the console.


### 🛠️ Example Usage: Predicting Customer Retention

```python
# Example Data: Features (e.g., hours spent on platform, number of purchases)
X = np.array([[5.0, 20.0], [2.0, 10.0], [8.0, 50.0], [1.0, 5.0]])  # Features
y = np.array([1, 0, 1, 0])  # Labels: 1 (retained), 0 (not retained)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X, y)

# Predict Retention for New Customers
X_new = np.array([[3.0, 15.0], [7.0, 30.0]])
y_pred = model.predict(X_new)

print("Predicted Retention (1 = Retained, 0 = Not Retained):", y_pred)
```

- This demonstrates how Logistic Regression can be applied to predict customer retention based on behavioral data, showcasing its practicality for real-world binary classification tasks.



--- 

## 🌟 Advantages 
   -  Simple and efficient for binary classification problems.

    - Outputs probabilities, allowing flexibility in decision thresholds.

    - Easily extendable to multiclass classification using the one-vs-rest (OvR) or multinomial approach.

## ⚠️ Limitations 

-  Assumes a linear relationship between features and log-odds of the target.

- Not effective when features are highly correlated or when there is a non-linear relationship.

## 🚀 Application 

=== "Application 1"
    **Medical Diagnosis**: Predicting the likelihood of a disease based on patient features.


=== "Application 2"
    **Marketing**: Determining whether a customer will purchase a product based on demographic and behavioral data.

