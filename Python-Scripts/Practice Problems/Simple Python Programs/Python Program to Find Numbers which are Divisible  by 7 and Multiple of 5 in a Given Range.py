# Python Program to Find Numbers which are Divisible 
# by 7 and Multiple of 5 in a Given Range

r = int(input("Enter the range: "))

for i in range(1,r+1):
    if (i%5==0 and i%7==0):
        print(i,end=", ")
