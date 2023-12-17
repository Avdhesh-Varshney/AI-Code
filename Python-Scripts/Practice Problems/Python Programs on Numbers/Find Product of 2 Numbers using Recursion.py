# Python Program to Find Product of 2 Numbers using Recursion

def prod(num1,num2,p):

    if(p>1):
        p = (num1*num2)
        return p

    return prod(num1,num2,p+1)

n1 = int(input("\nEnter 1st Number: "))
n2 = int(input("\nEnter 2nd Number: "))

print(f"\nProduct of {n1} and {n2} is {prod(n1,n2,1)}.\n")
