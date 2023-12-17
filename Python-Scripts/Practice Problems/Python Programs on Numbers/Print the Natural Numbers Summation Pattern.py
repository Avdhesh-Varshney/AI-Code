# Python Program to Read a Number n and Print the Natural Numbers Summation Pattern

n = int(input("Enter A Number: "))
sum = 0

for i in range(1,n):
    print(f"{i} + ",end="")
    sum = sum + i

print(f"{n} = {sum+n}")
