Aim: To predict housing prices using Linear Regression
Dataset: https://www.kaggle.com/datasets/ashydv/housing-dataset/data

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

HouseDF.info()

HouseDF.describe()

#display dataset columns
HouseDF.columns

#display data types of dataset columns
print(HouseDF.dtypes)

# EDA
sns.pairplot(HouseDF)

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,y_train)

print(lm.intercept_)


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
sns.distplot((y_test-predictions),bins=50);
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

#lower the errors, better the prediction model

