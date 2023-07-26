# Using Numpy, create an array 10x10 filled with number 255 and set the data type to float. Print this array to the console as shown below.

import numpy as np

# <---- Write your code here ---->

# Method 1
print(np.full(shape=(10, 10), fill_value=255, dtype=float))

# Method 2
print(np.ones(shape=(10, 10), dtype='float') * 255)
