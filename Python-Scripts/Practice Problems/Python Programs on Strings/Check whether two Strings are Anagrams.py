# Python Program to Check whether two Strings are Anagrams

s1 = input("\nEnter The First String: ").lower()
s2 = input("Enter The Second String: ").lower()

if(sorted(s1) == sorted(s2)):
    print(f"\n{s1} and {s2} are Anagrams.\n")
else:
    print(f"\n{s1} and {s2} are not Anagrams.\n")
