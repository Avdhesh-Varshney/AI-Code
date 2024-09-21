# Python Program to Check Whether the Entered Number is a Amicable Number or Not

def divisors(n):
    lis = []

    for i in range(1, n):
        if n%i == 0:
            lis.append(i)

    sum = 0
    for j in range(len(lis)):
        sum += lis[j]

    return sum

a = int(input("Enter Number A: "))
b = int(input("Enter Number B: "))

sum_a = divisors(a)
sum_b = divisors(b)

if(sum_a == b and sum_b == a):
    print(f"{a} & {b} are Amicable Numbers.\n")

else:
    print(f"{a} & {b} are not Amicable Numbers.\n")
