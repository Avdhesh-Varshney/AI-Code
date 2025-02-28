Aim: To predict housing prices using Linear Regression
Dataset: https://www.kaggle.com/datasets/ashydv/housing-dataset/data

Tech Stack



Language: Python

Libraries/Frameworks: pandas, numpy, seaborn, matplotlib, sklearn

Description: The project aims at predicting prices of houses based on features extracted from a real dataset set in Delhi, using Linear Regression. It performs basic Exploratory Data Analysis (EDA) in order to understand data trends.

This project can be a helpful tool for real estate decision-making by estimating property values, and helping market participants understand trends.

The approach of the project required selection and understanding the dataset foremost, visulaising the data and selecting releavnt and useful features, upon which a Linear Regression model was built and evaluated using metrics such as MAE, MSE, and RMSE.
Project Workflow and Basic code explanation:
Step 1 - Data Cleaning

Step 2 - Feature Engineering

Step 3 - Model Selection

Step 4 - Model Training

Step 5 - Evaluation

Step 6 - Deployment
#importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline
#read csv file
HouseDF = pd.read_csv("C:/Users/HP/Downloads/Housing.csv")

#display first 5 entries of dataset
HouseDF.head()

#dataset overview and feature details
HouseDF.info()

HouseDF.describe()

#display dataset columns
HouseDF.columns

#display data types of dataset columns
print(HouseDF.dtypes)

# EDA
sns.pairplot(HouseDF)

# pair plots to visualise features' relations
sns.distplot(HouseDF['price'])

#heatmap if converting non numeric values
HouseDF_encoded = HouseDF.copy()
HouseDF_encoded = pd.get_dummies(HouseDF_encoded, drop_first=True)
sns.heatmap(HouseDF_encoded.corr(), annot=True)

#heatmap if dropping non numeric values
drop non numeric columsn
sns.heatmap(HouseDF_numeric.corr(), annot=True)

#if dropping non numerical values
# Selecting numerical features as independent variables (X)
X = HouseDF[['area', 'bedrooms', 'bathrooms', 'stories', 'parking']]

# Selecting the target variable (y)
y = HouseDF['price']

#if using the non numerical values
# Encoding categorical variables
HouseDF_encoded = pd.get_dummies(HouseDF, drop_first=True)

# Selecting independent variables (X) - all numeric columns after encoding
X = HouseDF_encoded.drop(columns=['price'])

# Selecting the target variable (y)
y = HouseDF_encoded['price']


from sklearn.model_selection import train_test_split

# splitting dataset into training (60%) and test (40%) sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

# import linear regression model form scikit learn
from sklearn.linear_model import LinearRegression

# initialising the model
lm = LinearRegression()

# training model using the training data
lm.fit(X_train,y_train)

print(lm.intercept_)

# display feature coefficients
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df
predictions = lm.predict(X_test)
#visualisation of actual vs predicted values
plt.scatter(y_test,predictions)
sns.distplot((y_test-predictions),bins=50);
# import evaluation metrics
from sklearn import metrics

# display and calculation of the performance metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

#lower the errors, better the prediction model
Project Tradeoffs

Accuracy vs. Computation Time:
More features could improve accuracy but increase computational cost, addressed by selecting only highly correlated features.

Model Simplicity vs. Performance
Linear Regression for better explainability.
Screenshots of EDA

![Screenshot 2025-02-28 203414.png](<attachment:Screenshot 2025-02-28 203414.png>)

![Screenshot 2025-02-28 203431.png](<attachment:Screenshot 2025-02-28 203431.png>)

![Screenshot 2025-02-28 203437.png](<attachment:Screenshot 2025-02-28 203437.png>)
Evaluation Metrics for Linear Regression:

MAE: 798540.2157757834

MSE: 1233466174021.77

RMSE: 1110615.2232081864
Conclusion

House price is strongly influenced by area, stories, and bathrooms.

Features like airconditioning and parking also contribute significantly.
Use Cases

Real Estate Pricing, Market Trend Analysis
