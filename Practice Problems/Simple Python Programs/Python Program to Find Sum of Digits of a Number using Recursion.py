# Python Program to Find Sum of Digits of a Number using Recursion

def s_o_d(num):
    if(num==0):
        return 0
    return (int(num%10) + s_o_d(num/10))

n = int(input("Enter a number: "))

print(f"\nSum of all the digits of {n} number is: {s_o_d(n)}.\n")
