# Python Program to Find the Smallest Divisor of an Integer

n = int(input("Enter A Number: "))

for i in range(2,n+1):
    if(n%i==0):
        print(f"\n{i} is the Smallest Divisor of {n} after 1.\n")
        break
