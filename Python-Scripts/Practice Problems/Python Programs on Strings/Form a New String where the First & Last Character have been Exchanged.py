# Python Program to Form a New String where the First & Last Character have been Exchanged

s = input("\nEnter The String: ")

new = s[-1] + s[1:-1] + s[0]

print(f"\nNew String is: {new}\n")
