# Python Program to Check Whether a Given Number is Even or Odd Recursively

def check(n):
    if (n < 2):
        return (n % 2 == 0)
    return (check(n - 2))

n = int(input("Enter a number: "))
if(check(n)==True):
    print("\nNumber is even!\n")
else:
    print("\nNumber is odd!\n")
