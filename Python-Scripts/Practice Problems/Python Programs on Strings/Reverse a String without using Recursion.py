# Python Program to Reverse a String without using Recursion

s = input("\nEnter The String to be reversed: ")

print("\nReversed String = ",end="")

for i in reversed(range(len(s))):
    print(s[i],end="")

print("\n")
