# Python Program to Find the GCD of Two Numbers Using Recursion

def gcd(a,b):
    if(b == 0):
        return a
    else:
        return gcd(b,a%b)

n1 = int(input("\nEnter 1st value: "))
n2 = int(input("and 2nd value: "))

print(f"\nGCD/HCF of {n1} and {n2}: {gcd(n1,n2)}\n")
