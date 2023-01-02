# Python Program to Remove the Characters of Odd Index Values in a String

s = input("\nEnter The String: ")

lis = []

for i in range(len(s)):
    if(i%2 != 0):
        lis.append(s[i])

print(f"\nAfter Removing the Characters of Odd Index Values of String :- {s}\n")

for j in range(len(lis)):
    print(lis[j],end="")

print("\n")
