# Python Program to Find Whether a Number is a Power of Two

n = int(input("\nEnter A Number: "))

lis = []
countx = True

for i in range(2,n):
    if(n%i==0):
        lis.append(i)

for j in range(0,len(lis)):
    if(lis[j]%2!=0):
        countx = False

if(countx==True):
    print(f"\nYes, {n} is The Power of Two.\n")

else:
    print(f"\nNo, {n} is not The Power of Two.\n")
