# Python Program to Find the Sum of Sine Series --> sin(x) = x - x^3/3! + x^5/5! - x^7/7! + ...

import math

x = float(input("\nEnter the value of x in degrees: "))
n = int(input("Enter the number of terms: "))

countx = 2
p = 2
y = x*(22/(7*180))

if(n%2 == 0):
    for j in range(2,n-1):
        if(j%2 != 0):
            countx = countx + 1
else:
    for j in range(2,n-1):
        if(j%2 != 0):
            countx = countx + 1

print(f"\nsin({x}°) = {round(y,3)}",end="")

for j in range(2,n-1):
    if(j%2 != 0):
        if (p%2 == 0)and(p != countx):
            p = p + 1
            print(f" - {round(y,3)}^{j}/{j}!",end="")
        elif(p%2 != 0)and(p != countx):
            p = p + 1
            print(f" + {round(y,3)}^{j}/{j}!",end="")

if (n%2 == 0):
    if(countx%2==0):
        print(f" - {round(y,3)}^{n-1}/{n-1}!",end=" = ")
    else:
        print(f" + {round(y,3)}^{n-1}/{n-1}!",end=" = ")
else:
    if(countx%2==0):
        print(f" - {round(y,3)}^{n}/{n}!",end=" = ")
    else:
        print(f" + {round(y,3)}^{n}/{n}!",end=" = ")

s = 0

def sin(x,n):
    sine = 0
    for i in range(n):
        sign = (-1)**i
        pi=22/7
        y=x*(pi/180)
        sine = sine + ((y**(2.0*i+1))/math.factorial(2*i+1))*sign
    return sine

s = round(sin(x,n),3)

print(f"{s}\n")
