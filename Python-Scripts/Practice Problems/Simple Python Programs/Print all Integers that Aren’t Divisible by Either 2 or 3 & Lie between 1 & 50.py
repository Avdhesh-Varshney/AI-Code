# Python Program to Print all Integers that Aren’t Divisible 
# by Either 2 or 3 & Lie between 1 & 50

for i in range(1,51):
    if (i%2!=0 and i%3!=0):
        print(i,end=", ")
