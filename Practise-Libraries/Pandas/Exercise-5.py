'''Using the numpy and pandas create the following Series:

101    10.0
102    20.0
103    30.0
104    40.0
105    50.0
106    60.0
107    70.0
108    80.0
109    90.0
dtype: float64

In response, print this Series to the console.'''

import numpy as np
import pandas as pd

# <---- Write your code here ---->

# Method 1
x = np.arange(101, 110, 1)
y = np.arange(10.0, 100.0, 10.0)

s = pd.Series(y, x, float)

print(s)

# Method 2
s = pd.Series(
    data=np.arange(10, 100, 10), 
    index=np.arange(101, 110), 
    dtype='float'
)

print(s)
