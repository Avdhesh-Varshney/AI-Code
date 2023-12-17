# Python Program to Compute the Sum of Digits in a given Integer

num = int(input("Enter The Number: "))

sum = 0

l = int(len(str(num)))

if(num>0):
    n = num
else:
    n = (-1)*num

for i in range(0,l):
    m = int(n%10)
    n = int(n/10)
    sum = sum + m

print(f"The Sum of The Digits of {num} is: {sum}")
