# Python Program to Find the Sum of First N Natural Numbers

num = int(input("\nEnter A Number: "))
sum = 0

for i in range(1,num+1):
    sum = sum + i

print(f"\nSum of First {num} Natural Numbers are {sum}.\n")
