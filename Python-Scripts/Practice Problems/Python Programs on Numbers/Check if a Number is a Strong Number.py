# Python Program to Check if a Number is a Strong Number

num = int(input("\nEnter A Number: "))
n = num

sum = 0
l = int(len(str(n)))

lis = []

for i in range(1,l+1):
    if(n>0):
        m = n%10
        lis.append(m)
        n = int(n/10)

sum1 = 0

for j in range(0,l):
    q = 1

    for k in range(1,lis[j]+1):
        q = q*k

    sum1 = sum1 + q

if(sum1==num):
    print(f"\n{num} is a Strong Number.\n")

else:
    print(f"\n{num} is not a Strong Number.\n")
