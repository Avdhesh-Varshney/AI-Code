# Python Program to Remove the Duplicate Items from a List

lis1 = [1,4,6,7,3,6,5,2,7,56]
lis2 = [4,6,2,6,8,34,75,23,76,234,342]

lis = list(lis1 + lis2)
lis = list(set(lis))

print(f"\nAfter Removing the Duplicates Items from a List: {lis}\n")
