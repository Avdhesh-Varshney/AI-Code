# Python Program to Print all Numbers in a Range Divisible by a Given Number

r = int(input("Enter the range: "))

n = int(input("Enter the number by which the numbers are divisible: "))

for i in range(1,r+1):
    if(i%n==0):
        print(i,end=", ")
