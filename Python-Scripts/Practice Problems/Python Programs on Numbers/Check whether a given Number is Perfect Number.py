# Python Program to Check whether a given Number is Perfect Number

n = int(input("\nEnter A Number: "))
sum = 0

for i in range(1,n):
    if(n%i==0):
        sum = sum + i

if(sum==n):
    print(f"\n{n} is a Perfect Number.\n")

else:
    print(f"\n{n} is not a Perfect Number.\n")
