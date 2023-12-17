# Python Program to Check whether a given Number is Armstrong

num = int(input("\nEnter A Number: "))
n = num

l = int(len(str(n)))
lis = []

for i in range(0,l):
    if(n>0):
        m = n%10
        lis.append(m)
        n = int(n/10)

sum = 0

for j in range(0,l):
    sum = sum + (lis[j]*lis[j]*lis[j])

if(sum==num):
    print(f"\n{num} is a Armstrong Number.\n")

else:
    print(f"\n{num} is not a Armstrong Number.\n")
