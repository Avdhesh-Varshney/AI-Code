# Python Program to Find if a given Year is a Leap Year

n = int(input("Enter The Year: "))

if (n%4==0) and ((n%400==0) or (n%100!=0)):
    print(f"\n{n} is a Leap Year.\n")
else:
    print(f"\n{n} is not a Leap Year.\n")
