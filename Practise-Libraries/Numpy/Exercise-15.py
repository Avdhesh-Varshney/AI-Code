# Using Numpy create a two-dimensional array with the shape (10, 4) pseudo-randomly generated values from the standard normal distribution N(0,1). Set the random seed to 20. Print result to the console as shown below.

import numpy as np

# <---- Write your code here ---->

np.random.seed(20)

print(np.random.randn(10, 4))
