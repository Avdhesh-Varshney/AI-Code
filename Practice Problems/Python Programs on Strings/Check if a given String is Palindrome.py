# Python Program to Check if a given String is Palindrome

s = input("\nEnter The String: ")

cond = True

for i in range(int(len(s)/2)):
    if(s[i] != s[-1-i]):
        cond = False
        break

if(cond):
    print("\nGiven String is The Palindrome\n")
else:
    print("\nGiven String is not The Palindrome\n")
