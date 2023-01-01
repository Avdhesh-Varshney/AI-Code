# Python Program to Find the GCD of Two Numbers

n1 = int(input("\nEnter 1st value: "))
n2 = int(input("and 2nd value: "))

p = 1

for i in range(1, max(n1, n2)):
    if(n1%i == 0) and (n2%i == 0):
        p = i

print(f"\nGCD/HCF of {n1} and {n2}: {p}\n")
