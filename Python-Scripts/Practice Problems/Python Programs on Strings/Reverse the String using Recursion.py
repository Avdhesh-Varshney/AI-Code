# Python Program to Reverse the String using Recursion

def reverse(string):
    if len(string) == 0:
        return string
    else:
        return reverse(string[1:]) + string[0]

s = input("\nEnter The String to be reversed: ")

print(f"\nReversed String = {reverse(s)}\n")
