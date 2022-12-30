# Python Program to Print an Identity Matrix

n = int(input("\nEnter The Dimension of Identity Matrix: "))

print(f"\n{n}x{n} Matrix is: \n")

for i in range(0,n):
    for j in range(0,n):
        if(i==j):
            print(1,end="  ")
        else:
            print(0,end="  ")
    print("\n")
