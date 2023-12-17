# Python Program to Check Common Letters in Two Input Strings

s1 = input("\nEnter First String: ").lower()
s2 = input("Enter Second String: ").lower()

a = list(set(s1) & set(s2))

countx = 0

print("\nThe common letters are: ",end = "")

for i in a:
    print(i,end = " ")
    countx += 1

print(f"\nTotal Number of Common Letters are {countx}\n")
