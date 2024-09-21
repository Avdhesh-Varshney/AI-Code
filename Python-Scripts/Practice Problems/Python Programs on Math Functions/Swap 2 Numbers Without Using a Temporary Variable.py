# Python Program to Swap 2 Numbers Without Using a Temporary Variable

n1 = int(input("\nEnter Value of n1: "))
n2 = int(input("Enter Value of n2: "))

n1 = n1 + n2
n2 = n1 - n2
n1 = n1 - n2

print(f"\nAfter Swaping Values are :\nn1= {n1}, n2= {n2}\n")
