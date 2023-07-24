# Check if all elements from the following arrays return the logical value True

import numpy as np

A = np.array([[3, 2, 1, 4],
              [5, 2, 1, 6]])

B = np.array([[3, 2, 1, 4],
              [5, 2, 0, 6]])

C = np.array([[True, False, False],
              [True, True, True]])

D = np.array([0.1, 0.3])

# <---- Write your code here ---->

# Method 1

print("A: ", end="")
print(np.all(A))

print("B: ", end="")
print(np.all(B))

print("C: ", end="")
print(np.all(C))

print("D: ", end="")
print(np.all(D))

# Method 2

for name, array in zip(list('ABCD'), [A, B, C, D]):
    print(f'{name}: {np.all(array)}')
