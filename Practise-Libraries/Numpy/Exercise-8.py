# Check which numbers (element-wise) from the A array are greater than numbers from the B array and print result to the console as shown below.

import numpy as np

A = np.array([0.4, 0.5, 0.3, 0.9])
B = np.array([0.38, 0.51, 0.3, 0.91])

# <---- Write your code here ---->

# Method 1
print(A > B)

# Method 2
print(np.greater(A, B))
