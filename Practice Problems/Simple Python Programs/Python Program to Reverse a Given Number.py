# Python Program to Reverse a Given Number

num = int(input("Enter a number: "))

l = int(len(str(num)))
a = num
lis = []

for i in range(0,l):
    n = int(a%10)
    a = a/10
    lis.append(n)

new_num = 0
for j in range(0,l):
    m = lis[j]*pow(10,l-1-j)
    new_num = new_num + m

print(f"\nReverse of {num} is: {new_num}\n")
