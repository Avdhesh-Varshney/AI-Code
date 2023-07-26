# Check if the following arrays are equal (element-wise)

import numpy as np

A = np.array([0.4, 0.5, 0.3])
B = np.array([0.39999999, 0.5000001, 0.3])

# <---- Write your code here ---->

print(np.allclose(A, B))
