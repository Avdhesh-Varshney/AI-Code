# STANDARD SCALER 

A custom implementation of a StandardScaler class for scaling numerical data in a pandas DataFrame or NumPy array. The class scales the features to have zero mean and unit variance.

## Features

- **fit**: Calculate the mean and standard deviation of the data.
- **transform**: Scale the data to have zero mean and unit variance.
- **fit_transform**: Fit the scaler and transform the data in one step.
- **get_params**: Retrieve the mean and standard deviation calculated during fitting.

## Methods

1. `__init__(self)`
    - Initializes the StandardScaling class.
    - No parameters are required.
2. `fit(self, data)`
    - Calculates the mean and standard deviation of the data.
    - Parameters:
        - data (pandas.DataFrame or numpy.ndarray): The data to fit.
    - Raises:
        - TypeError: If the input data is not a pandas DataFrame or NumPy array.
3. `transform(self, data)`
    - Transforms the data to have zero mean and unit variance.
    - Parameters:
        - data (pandas.DataFrame or numpy.ndarray): The data to transform.
    - Returns:
        - numpy.ndarray: The scaled data.
    - Raises:
        - Error: If transform is called before fit or fit_transform.
4. `fit_transform(self, data)`
    - Fits the scaler to the data and transforms the data in one step.
    - Parameters:
        - data (pandas.DataFrame or numpy.ndarray): The data to fit and transform.
    - Returns:
        - numpy.ndarray: The scaled data.
5. `get_params(self)`
    - Retrieves the mean and standard deviation calculated during fitting.
    - Returns:
        - dict: Dictionary containing the mean and standard deviation.
    - Raises:
        - Error: If get_params is called before fit.

## Error Handling

- Raises a TypeError if the input data is not a pandas DataFrame or NumPy array in the fit method.
- Raises an error if transform is called before fit or fit_transform.
- Raises an error in get_params if called before fit.

## Use Case

![use_case](https://github.com/user-attachments/assets/857aa25a-6cb9-4320-aa43-993bd289bd32)

## Output

![output](https://github.com/user-attachments/assets/ea2c1374-78c7-4cff-a431-eced068c052f)


- standard_scaler.py file 

```py
import pandas as pd
import numpy as np

# Custom MinMaxScaler class
class StandardScaling:
    # init function
    def __init__(self):     
        self.data_mean_ = None
        self.data_std_ = None

    # fit function to calculate min and max value of the data
    def fit(self, data):
        # type check
        if not (type(data)==pd.DataFrame or type(data)==np.ndarray):
            raise f"TypeError : parameter should be a Pandas.DataFrame or Numpy.ndarray; {type(data)} found"
        elif type(data)==pd.DataFrame:
            data = data.to_numpy()
        
        self.data_mean_ = np.mean(data, axis=0)
        self.data_std_ = np.sqrt(np.var(data, axis=0))
    
    # transform function
    def transform(self, data):
        if self.data_mean_ is None or self.data_std_ is None:
            raise "Call StandardScaling.fit() first or call StandardScaling.fit_transform() as the required params not found"
        else:
            data_scaled = (data - self.data_mean_) / (self.data_std_)
            return data_scaled

    # fit_tranform function
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    
    # get_params function
    def get_params(self):
        if self.data_mean_ is None or self.data_std_ is None:
            raise "Params not found! Call StandardScaling.fit() first"
        else:
            return {"Mean" : self.data_mean_,
                    "Standard Deviation" : self.data_std_}
```

- test_standard_scaler.py file 

```py
import os
import sys
# for resolving any path conflict
current = os.path.dirname(os.path.realpath("standard_scaler.py"))
parent = os.path.dirname(current)
sys.path.append(current)

import pandas as pd

from Standard_Scaler.standard_scaler import StandardScaling

# Example DataFrame
data = {
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50],
    'C': [100, 200, 300, 400, 500]
}

df = pd.DataFrame(data)

# Initialize the CustomMinMaxScaler
scaler = StandardScaling()

# Fit the scaler to the data and transform the data
scaled_df = scaler.fit_transform(df)

print("Original DataFrame:")
print(df)
print("\nScaled DataFrame:")
print(scaled_df)
print("\nAssociated Parameters:")
print(scaler.get_params())
```