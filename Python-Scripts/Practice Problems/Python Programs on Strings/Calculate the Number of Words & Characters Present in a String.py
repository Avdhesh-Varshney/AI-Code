# Python Program to Calculate the Number of Words & Characters Present in a String

s = input("\nEnter a String: ")
ch = input("\nEnter a Word/Character: ")

countx = 0

if(len(ch)==1):
    for i in range(len(s)):
        if(ch==s[i]):
            countx += 1
else:
    words = s.split()
    for w in words:
        if w == ch:
            countx += 1

print(f"\n{ch} is found {countx} times.\n")
