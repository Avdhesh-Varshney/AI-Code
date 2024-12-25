# ORDINAL ENCODER 

A custom implementation of an OrdinalEncoder class for encoding categorical data into ordinal integers using a pandas DataFrame. The class maps each unique category to an integer based on the order of appearance.

## Features

- **fit**: Learn the mapping of categories to ordinal integers for each column.
- **transform**: Transform the categorical data to ordinal integers based on the learned mapping.
- **fit_transform**: Fit the encoder and transform the data in one step.

## Methods

1. `__init__(self)`
    - Initializes the OrdinalEncoding class.
    - No parameters are required.
2. `fit(self, data)`
    - Learns the mapping of categories to ordinal integers for each column.
    - Parameters:
        - data (pandas.DataFrame): The data to fit.
    - Raises:
        - TypeError: If the input data is not a pandas DataFrame.
3. `transform(self, data)`
    - Transforms the categorical data to ordinal integers based on the learned mapping.
    - Parameters:
        - data (pandas.DataFrame): The data to transform.
    - Returns:
        - pandas.DataFrame: The transformed data.
    - Raises:
        - Error: If transform is called before fit or fit_transform.
4. `fit_transform(self, data)`
    - Fits the encoder to the data and transforms the data in one step.
    - Parameters:
        - data (pandas.DataFrame): The data to fit and transform.
    - Returns:
        - pandas.DataFrame: The transformed data.

## Error Handling

- Raises a TypeError if the input data is not a pandas DataFrame in the fit method.
- Raises an error if transform is called before fit or fit_transform.

## Use Case

![use_case](https://github.com/user-attachments/assets/af3f20f7-b26a-45b7-9a0f-fc9dcdd99534)

## Output

![output](https://github.com/user-attachments/assets/12f31b6b-c165-460f-b1e9-5726663f625d)


- ordinal_encoder.py file 

```py
import pandas as pd

class OrdinalEncoding:
    def __init__(self):
        self.category_mapping = {}
    
    def fit(self, data):
        # Fit the encoder to the data (pandas DataFrame).
        # type check
        if not type(data)==pd.DataFrame:
            raise f"Type of data should be Pandas.DataFrame; {type(data)} found"
        for column in data.columns:
            unique_categories = sorted(set(data[column]))
            self.category_mapping[column] = {category: idx for idx, category in enumerate(unique_categories)}
    
    def transform(self, data):
        # Transform the data (pandas DataFrame) to ordinal integers.
        # checking for empty mapping
        if not self.category_mapping:
            raise "Catrgorical Mapping not found. Call OrdinalExcoding.fit() method or call OrdinalEncoding.fit_transform() method"
        
        data_transformed = data.copy()
        for column in data.columns:
            data_transformed[column] = data[column].map(self.category_mapping[column])
        return data_transformed
    
    def fit_transform(self, data):
        # Fit the encoder and transform the data in one step.
        self.fit(data)
        return self.transform(data)
```

- test_ordinal_encoder.py file 

```py
import os
import sys
# for resolving any path conflict
current = os.path.dirname(os.path.realpath("ordinal_encoder.py"))
parent = os.path.dirname(current)
sys.path.append(current)

import pandas as pd

from Ordinal_Encoder.ordinal_encoder import OrdinalEncoding

# Example usage
data = {
    'Category1': ['low', 'medium', 'high', 'medium', 'low', 'high', 'medium'],
    'Category2': ['A', 'B', 'A', 'B', 'A', 'B', 'A'],
    'Category3': ['X', 'Y', 'X', 'Y', 'X', 'Y', 'X']
}
df = pd.DataFrame(data)

encoder = OrdinalEncoding()
encoded_df = encoder.fit_transform(df)

print("Original DataFrame:")
print(df)
print("\nEncoded DataFrame:")
print(encoded_df)
```
