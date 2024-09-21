# Python Program to Find out the Roots of a Quadratic Equation

import math

# Taking Input
print("\nEnter The Coefficients of ax^2+bx+c=0: ")
a = float(input("a= "))
b = float(input("b= "))
c = float(input("c= "))

# Calcuating Roots
dis = (b*b)-4*a*c

if(dis>=0):
    x1 = (-b/(2*a)) + (math.sqrt(dis)/(2*a))
    x2 = (-b/(2*a)) - (math.sqrt(dis)/(2*a))

else:
    x1 = f"{(-b/(2*a))} + i{(math.sqrt(-1*dis)/(2*a))}"
    x2 = f"{(-b/(2*a))} + i{(math.sqrt(-1*dis)/(2*a))}"

# Printing Statements
if(b>0):
    print(f"\nRoots of {a}x^2 + {b}x + {c}= 0 are {x1}, and {x2}.\n")

else:
    print(f"\nRoots of {a}x^2 - {-1*b}x + {c}= 0 are {x1}, and {x2}.\n")
