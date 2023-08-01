# Set the random seed to 10. Then using Numpy create a one-dimensional array consisting of 30 pseudo-randomly generated values from the uniform distribution [0,1). Print result to the console as shown below.

import numpy as np

# <---- Write your code here ---->

np.random.seed(10)

# Method 1
print(np.random.rand(30))

# Method 2
print(np.random.random(30))
