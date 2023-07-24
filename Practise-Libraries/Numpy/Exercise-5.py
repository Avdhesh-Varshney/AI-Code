# Check if the following array has missing data (np.nan)

import numpy as np

A = np.array([[3, 2, 1, np.nan],
              [5, np.nan, 1, 6]])

# <---- Write your code here ---->

print(np.isnan(A))
