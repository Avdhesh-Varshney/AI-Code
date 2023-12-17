# Python Program to Merge Two Lists and Sort it

lis1 = [1,4,6,7,3,6,5,2,7,56]
lis2 = [4,6,2,6,8,34,75,23,76,234,342]

lis = list(lis1 + lis2)

newlis = sorted(lis)

print(f"\nRequired List: {newlis}\n")
