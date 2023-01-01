# Python Program to Find the Sum of the Series: 1 + x^2/2 + x^3/3 + … x^n/n

x = float(input("\nEnter The value of x: "))
n = int(input("Enter The value of n: "))

s = 1

print("\nSum of The Series: \n1 +",end=" ")

for a in range(2,n):
    print(f"{x}^{a}/{a}",end=" + ")

print(f"{x}^{n}/{n}",end=" = ")

for i in range(2,n+1):
    s = s + (pow(x,i))/i

print(f"{s}\n")
