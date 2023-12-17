# Python Program to Check if a Number is a Palindrome

num = int(input("Enter a number: "))

l = len(str(num))
lis = []
m = num

for i in range(0,l):
    n = m%10
    m = int(m/10)
    lis.append(n)

countx = 0
k = l-1
for j in range(0,int(l/2)):
    if (lis[j] == lis[k]):
        countx = countx + 1
        k = k - 1

if (countx == int(l/2)):
    print("Entered Number is a Palindrome.")
else:
    print("Entered Number is not a Palindrome.")
