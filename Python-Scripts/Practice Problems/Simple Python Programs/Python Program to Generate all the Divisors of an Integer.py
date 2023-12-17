# Python Program to Generate all the Divisors of an Integer

n = int(input("Enter a number: "))

print("\nAll The Divisors are: ",end="")

for i in range(1,n+1):
    if(n%i == 0):
        print(i,end=", ")

print("\n")
