# Check if the following arrays are equal (element-wise)

import numpy as np

A = np.array([0.4, 0.5, 0.3])
B = np.array([0.3999999999, 0.5000000001, 0.3])

# <---- Write your code here ---->

# Method 1
print(A == B)

# Method 2
print(np.equal(A, B))
