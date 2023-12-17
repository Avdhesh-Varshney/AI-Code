# Python Program to Find the Length of a List Using Recursion

def length(lst):
    if not lst:
        return 0
    return 1 + length(lst[1::2]) + length(lst[2::2])

a = [1,34,65,34,65,23,78,45,8,23,24,34,8,8]

print(f"\nLength of the string is: {length(a)}\n")
