# Python Program to Print Numbers in a Range (1,upper) Without Using any Loops

def prnt(n):
    if(n>0):
        prnt(n-1)
        print(n,end=", ")

num = int(input("\nEnter The Upper Limit: "))

print("\n")

prnt(num)
