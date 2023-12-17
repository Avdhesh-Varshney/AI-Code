# Python Program to Check if the Substring is present in the given String

s = input("\nEnter The Main String: ").lower()
sb = input("Enter The Sub-String: ").lower()

words = s.split()
swords = sb.split()

cond = 0

for i in range(len(words)):
    for j in range(len(swords)):
        if(swords[j]==words[i]):
            cond += 1

if(cond==len(swords)):
    print("\nYes, Sub-string is present in The Main String\n")
else:
    print("\nNo, Sub-string is not present in The Main String\n")
