# Python Program to Remove the nth Index Character from a Non-Empty String

s = input("\nEnter The String: ")
n = int(input("Enter The Index: "))

if(n < len(s)):
    s = s.replace(s[n-1], "")
    print(f"\nNew String is:- {s}.\n")
else:
    print("\nYou are entering index out of range.\n")
