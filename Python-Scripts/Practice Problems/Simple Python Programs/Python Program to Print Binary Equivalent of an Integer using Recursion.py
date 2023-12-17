# Python Program to Print Binary Equivalent of an Integer 
# using Recursion

def binary(n):
    if n > 1:
       binary(n//2)
    print(n%2,end="")

num = int(input("Enter A Number: "))

if (num >= 0):
    binary(num)
else:
    binary((-1)*num)
