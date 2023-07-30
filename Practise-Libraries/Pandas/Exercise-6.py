# Convert its type to int and print this Series to the console.

import pandas as pd


series = pd.Series(['001', '002', '003', '004'], list('abcd'))

# <---- Write your code here ---->

# Method 1
series = pd.to_numeric(series)
print(series)

# Method 2
series = series.astype(int)
print(series)
