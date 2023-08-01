# Using Numpy create a 6x6 two-dimensional array - identity matrix with int data type. Print this array to the console as shown below.

import numpy as np

# <---- Write your code here ---->

# Method 1
print(np.identity(6, dtype='int'))

# Method 2
print(np.eye(6, 6, dtype='int'))
