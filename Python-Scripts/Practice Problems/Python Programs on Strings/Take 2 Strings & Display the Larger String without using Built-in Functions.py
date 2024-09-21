# Python Program to Take 2 Strings & Display the Larger String without using Built-in Functions

s1 = input("\nEnter The First String: ")
s2 = input("Enter The Second String: ")

c1 = 0
c2 = 0

for i in s1:
    c1 += 1

for j in s2:
    c2 += 1

if(c1 == c2):
    print(f"\nBoth The Strings are of same Length.\n")
elif(c1 > c2):
    print(f"\n{s1} is the Larger String\n")
else:
    print(f"\n{s2} is the Larger String\n")
