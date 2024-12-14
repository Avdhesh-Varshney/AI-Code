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