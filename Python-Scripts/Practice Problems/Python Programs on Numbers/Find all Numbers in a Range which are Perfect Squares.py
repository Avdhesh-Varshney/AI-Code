# Python Program to Find all Numbers in a Range which are Perfect Squares and Sum of all Digits in the Number is Less than 10

r = int(input("\nEnter The Range: "))

lis = []

for i in range(1,r):
    if(i*i < r):
        lis.append(i*i)

sum = 0

for j in range(0,len(lis)):
    if(lis[j]<10):
        sum = sum + lis[j]

print(f"\nAll The Perfect Squares in a given range are {lis}\n")

print(f"\nSum of all Digits in the Number is less than 10 is {sum}\n")
