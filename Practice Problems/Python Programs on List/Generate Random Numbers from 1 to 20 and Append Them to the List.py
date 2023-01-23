# Python Program to Generate Random Numbers from 1 to 20 and Append Them to the List

import random

n = int(input("\nHow many times you want to generate a random number: "))

lis = []

for i in range(n):
    lis.append(random.randint(1,21))

print(f"\nRequired List: {lis}\n")
