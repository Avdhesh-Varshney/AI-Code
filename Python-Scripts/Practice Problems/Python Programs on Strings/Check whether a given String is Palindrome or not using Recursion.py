# Python Program to Check whether a given String is Palindrome or not using Recursion

def is_palindrome(s):
    if len(s) < 1:
        return True
    else:
        if s[0] == s[-1]:
            return is_palindrome(s[1:-1])
        else:
            return False

s = str(input("\nEnter The string: "))

if(is_palindrome(s)):
    print("\nString is a palindrome!\n")
else:
    print("\nString isn't a palindrome!\n")
