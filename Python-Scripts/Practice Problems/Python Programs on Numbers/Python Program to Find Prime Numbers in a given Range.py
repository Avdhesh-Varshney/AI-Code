# Python Program to Find Prime Numbers in a given Range

n = int(input("\nEnter The Range: "))
lis = []

if (n == 2):
    print("\n2 is a Prime Number.\n")

elif(n<=1):
    print("\nIn this range, There is not any Prime Number.\n")

else:
    print("\nPrime Numbers are: ",end="")
    for i in range(3,n+1):
        countx = 0

        for j in range(2,i-1):
            if(i%j==0):
                countx = 1

        lis.append(countx)

for k in range(0,n-2):
    if(lis[k]==0):
        print(k+3,end=", ")

print("\n")
