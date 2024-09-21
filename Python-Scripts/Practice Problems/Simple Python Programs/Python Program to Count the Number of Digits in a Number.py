# Python Program to Count the Number of Digits in a Number

n = int(input("Enter a number: "))

countx = 0

while(n>0):
    countx = countx + 1
    n = int(n/10)

print(f"Total number of digits are: {countx}")
