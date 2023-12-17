# Python Program to Count the Number of Vowels in a String

s = input("\nEnter The String: ")

lis = ['a','A','e','E','i','I','o','O','u','U']

countx = 0

for i in range(len(s)):
    for j in range(len(lis)):
        if(s[i]==lis[j]):
            countx += 1

print(f"\n{countx} Vowels are Found in the Given String\n")
