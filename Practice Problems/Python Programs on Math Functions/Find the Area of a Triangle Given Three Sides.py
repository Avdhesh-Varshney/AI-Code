# Python Program to Find the Area of a Triangle Given Three Sides

import math

print("\nEnter The Sides of Triangle: \n")
a = float(input("First Side: "))
b = float(input("Second Side: "))
c = float(input("Third Side: "))

s = (a+b+c)/2

r = s*(s-a)*(s-b)*(s-c)

area = math.sqrt(r)

print(f"\nArea of Triangle is: {area}.\n")
