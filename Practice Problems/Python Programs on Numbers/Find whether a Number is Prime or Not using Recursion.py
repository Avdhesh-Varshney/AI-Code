# Python Program to Find whether a Number is Prime or Not using Recursion

def isprime(num,i=2):
    if(num<=2):
        return True if (n==2) else False
    elif(n % i == 0):
        return False

    if (i * i > n):
        return True

    return isprime(n, i+1)

n = int(input("Enter A Number: "))

if (isprime(n)):
    print("Prime Number")

else:
    print("Not Prime Number")
