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