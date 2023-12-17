# Python Program that Displays which Letters are in the First String but not in the Second

s1 = input("\nEnter First String: ").lower()
s2 = input("Enter Second String: ").lower()

a = list(set(s1) - set(s2))

print("\nThe common letters are: ",end = "")

for i in a:
    print(i,end = " ")

print("\n")
