# Python Program to Count Number of Lowercase Characters in a String

s = input("\nEnter The String: ")

countx = 0

for i in range(len(s)):
    for j in range(1,27):
        if(s[i]==chr(96+j)):
            countx += 1

print(f"\n{countx} Lowercase Characters are Found!\n")
