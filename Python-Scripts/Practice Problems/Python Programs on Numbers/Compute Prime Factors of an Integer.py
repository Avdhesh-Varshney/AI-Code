# Python Program to Compute Prime Factors of an Integer

n = int(input("\nEnter A Number: "))

lis = []
lis0 = []

countx = 1

for i in range(2,n):
    if(n%i==0):
        lis0.append(i)

        for j in range(2,i):
            if(i%j==0):
                countx = 0
                lis.append(i)

test = list(set(lis))

for k in range(0,len(test)):
    lis0.remove(test[k])

print(f"\nPrime Factors of {n}: {lis0}\n")
