# MIN MAX SCALER 

A custom implementation of a MinMaxScaler class for scaling numerical data in a pandas DataFrame. The class scales the features to a specified range, typically between 0 and 1.

## Features

- **fit**: Calculate the minimum and maximum values of the data.
- **transform**: Scale the data to the specified feature range.
- **fit_transform**: Fit the scaler and transform the data in one step.
- **get_params**: Retrieve the minimum and maximum values calculated during fitting.

## Methods

1. `__init__(self, feature_range=(0, 1))`
    - Initializes the MinMaxScaling class.
    - Parameters:
        - feature_range (tuple): Desired range of transformed data. Default is (0, 1).
2. `fit(self, data)`
    - Calculates the minimum and maximum values of the data.
    - Parameters:
        - data (pandas.DataFrame): The data to fit.
3. `transform(self, data)`
    - Transforms the data to the specified feature range.
    - Parameters:
        - data (pandas.DataFrame): The data to transform.
    - Returns:
        - pandas.DataFrame: The scaled data.
4. `fit_transform(self, data)`
    - Fits the scaler to the data and transforms the data in one step.
    - Parameters:
        - data (pandas.DataFrame): The data to fit and transform.
    - Returns:
        - pandas.DataFrame: The scaled data.
5. `get_params(self)`
    - Retrieves the minimum and maximum values calculated during fitting.
    - Returns:
        - dict: Dictionary containing the minimum and maximum values.

## Error Handling

- Raises a TypeError if the input data is not a pandas DataFrame in the fit method.
- Raises an error if transform is called before fit or fit_transform.
- Raises an error in get_params if called before fit.

## Use Case

![use_case](https://github.com/user-attachments/assets/86cc2962-e744-490d-97a6-c084496701de)

## Output

![output](https://github.com/user-attachments/assets/d62b9856-d67c-4c92-a2db-f0e76409856a)


- min_max_scaler.py file 

```py
import pandas as pd

# Custom MinMaxScaler class
class MinMaxScaling:
    # init function
    def __init__(self, feature_range=(0, 1)):  # feature range can be specified by the user else it takes (0,1)
        self.min = feature_range[0]
        self.max = feature_range[1]
        self.data_min_ = None
        self.data_max_ = None

    # fit function to calculate min and max value of the data
    def fit(self, data):
        # type check
        if not type(data)==pd.DataFrame:
            raise f"TypeError : parameter should be a Pandas.DataFrame; {type(data)} found"
        else:
            self.data_min_ = data.min()
            self.data_max_ = data.max()
    
    # transform function
    def transform(self, data):
        if self.data_max_ is None or self.data_min_ is None:
            raise "Call MinMaxScaling.fit() first or call MinMaxScaling.fit_transform() as the required params not found"
        else:
            data_scaled = (data - self.data_min_) / (self.data_max_ - self.data_min_)
            data_scaled = data_scaled * (self.max - self.min) + self.min
            return data_scaled

    # fit_tranform function
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    
    # get_params function
    def get_params(self):
        if self.data_max_ is None or self.data_min_ is None:
            raise "Params not found! Call MinMaxScaling.fit() first"
        else:
            return {"Min" : self.data_min_,
                    "Max" : self.data_max_}
```

- test_min_max_scaler.py file 

```py
import os
import sys
# for resolving any path conflict
current = os.path.dirname(os.path.realpath("min_max_scaler.py"))
parent = os.path.dirname(current)
sys.path.append(current)

import pandas as pd

from Min_Max_Scaler.min_max_scaler import MinMaxScaling

# Example DataFrame
data = {
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50],
    'C': [100, 200, 300, 400, 500]
}

df = pd.DataFrame(data)

# Initialize the CustomMinMaxScaler
scaler = MinMaxScaling()

# Fit the scaler to the data and transform the data
scaled_df = scaler.fit_transform(df)

print("Original DataFrame:")
print(df)
print("\nScaled DataFrame:")
print(scaled_df)
```
