# Python Program to Find the LCM of Two Numbers Using Recursion

def lcm(a, b):
    lcm.multiple = lcm.multiple + b
    if((lcm.multiple % a == 0) and (lcm.multiple % b == 0)):
        return lcm.multiple
    else:
        lcm(a, b)
    return lcm.multiple

lcm.multiple = 0

a = int(input("\nEnter 1st value: "))
b = int(input("and 2nd value: "))

if(a > b):
    print(f"\nLCM of {a} and {b}: {lcm(b,a)}\n")
else:
    print(f"\nLCM of {a} and {b}: {lcm(a,b)}\n")
