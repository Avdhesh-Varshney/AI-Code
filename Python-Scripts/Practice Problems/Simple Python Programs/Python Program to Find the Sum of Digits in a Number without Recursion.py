# Python Program to Find the Sum of Digits in a Number without Recursion

n = int(input("Enter a Number: "))

sum = 0

while(n>0):
    sum = sum + int(n%10)
    n = n/10

print(f"\nSum of all the digits of Entered number is {sum}.\n")
