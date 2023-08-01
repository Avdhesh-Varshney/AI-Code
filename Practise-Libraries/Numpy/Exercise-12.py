'''Using Numpy create the following array:

[[10 11 12 13 14 15 16 17 18 19]
 [20 21 22 23 24 25 26 27 28 29]
 [30 31 32 33 34 35 36 37 38 39]
 [40 41 42 43 44 45 46 47 48 49]
 [50 51 52 53 54 55 56 57 58 59]
 [60 61 62 63 64 65 66 67 68 69]
 [70 71 72 73 74 75 76 77 78 79]
 [80 81 82 83 84 85 86 87 88 89]
 [90 91 92 93 94 95 96 97 98 99]]

Note that the shape of this array is (9, 10). In response, print array to the console.'''

import numpy as np

# <---- Write your code here ---->

arr = np.arange(10, 100, 1, dtype='int')

# Method 1
arr = arr.reshape(9, -1)

# Method 2
arr = arr.reshape(9, 10)

# Method 3
arr = arr.reshape(-1, 10)

print(arr)
