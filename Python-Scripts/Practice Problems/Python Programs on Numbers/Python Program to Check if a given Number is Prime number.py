# Python Program to Check if a given Number is Prime number

n = int(input("Enter a number: "))
countx = 0

if(n<=1):
    print(f"\n{n} is not a Prime Number.\n")

else:
    for i in range(2,n):
        if(n%i==0):
            countx = 1

    if(countx==0):
        print(f"\n{n} is a Prime Number.\n")

    else:
        print(f"\n{n} is not a Prime Number.\n")
