# for any import conflict
import os
import sys
current = os.path.dirname("ordinal_encoder.py")
parent = os.path.dirname(current)
sys.path.append(current)

import pandas as pd

# importing custom class
from Ordinal_Encoding.ordinal_encoder import OrdinalEncoding

data = {
    'color': ['red', 'green', 'blue', 'green', 'red'],
    'size': ['S', 'M', 'L', 'M', 'S'],
    'class': ['A', 'B', 'A', 'B', 'A']
}
df = pd.DataFrame(data)

print(f"Original Data : \n{df}")

# Initialize the custom OrdinalEncoder
encoder = OrdinalEncoding()

# Fit the encoder to the data and transform it
try:
    encoded_df = encoder.fit_transform(df)
    print("\nEncoded Data:")
    print(encoded_df)
except ValueError as e:
    print(f"\nException: {e}")