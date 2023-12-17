# Python Program to Find the Sum of Cosine Series --> cos(x) = 1 - x^2/2! + x^4/4! - x^6/6! + ...

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

print(f"\ncos({x}°) = 1",end="")

for j in range(2,n-1):
    if(j%2 == 0):
        if (p%2 == 0)and(p != countx):
            p = p + 1
            print(f" - {round(y,3)}^{j}/{j}!",end="")
        elif(p%2 != 0)and(p != countx):
            p = p + 1
            print(f" + {round(y,3)}^{j}/{j}!",end="")

if (n%2 == 0):
    if(countx%2==0):
        print(f" - {round(y,3)}^{n}/{n}!",end=" = ")
    else:
        print(f" + {round(y,3)}^{n}/{n}!",end=" = ")
else:
    if(countx%2==0):
        print(f" - {round(y,3)}^{n-1}/{n-1}!",end=" = ")
    else:
        print(f" + {round(y,3)}^{n-1}/{n-1}!",end=" = ")

s = 0

def cosine(x,n):
    cosx = 1
    sign = -1
    for i in range(2, n, 2):
        pi=22/7
        y=x*(pi/180)
        cosx = cosx + (sign*(y**i))/math.factorial(i)
        sign = -sign
    return cosx

s = round(cosine(x,n),3)

print(f"{s}\n")
