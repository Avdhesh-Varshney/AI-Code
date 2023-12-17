# Python Program to Find the Binary Equivalent of a Number 
# without Using Recursion

n = int(input("Enter A Number: "))

lis = []

while(n>0):
    if(n%2==0):
        lis.append(0)
        n = int(n/2)
    else:
        lis.append(1)
        n = int(n/2)

for i in range(len(lis) - 1, -1, -1):
    print(lis[i], end = "")
