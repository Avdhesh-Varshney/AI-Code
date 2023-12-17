# Python Program to Compute a Polynomial Equation

import math

print("\nEnter the coefficients of the form ax^3 + bx^2 + cx + d: \n")
lst = []

for i in range(0,4):
    a = int(input(f"Enter the value of {chr(96+i+1)}: "))
    lst.append(a)

x = int(input("\nEnter the value of x: "))
sum1 = 0

j = 3
for i in range(0,3):
    while(j>0):
        sum1 = sum1 + (lst[i]*math.pow(x,j))
        break
    j = j-1

sum1 = sum1 + lst[3]

print(f"\nThe value of the polynomial is {sum1}\n")
