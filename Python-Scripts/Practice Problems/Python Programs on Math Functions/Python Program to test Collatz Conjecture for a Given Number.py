# Python Program to test Collatz Conjecture for a Given Number

def collatz(n):

    while(n > 1):
        print(n,end=" ")

        if (n % 2):
            n = 3*n + 1

        else:
            n = n//2

    print(1, end=" ") 

n = int(input("\nEnter n: "))

print("\nSequence: ", end="")

collatz(n)

print("\n")
