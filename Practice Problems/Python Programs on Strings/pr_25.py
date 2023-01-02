# Python program that accepts a hyphen-separated sequence of words as input and 
# prints the words in a hyphen-separated sequence after sorting them alphabetically

print("\nEnter The Sequence of Words: ",end="")

items = [n for n in input().split('-')]

items.sort()

print("\nRequired Answer is: ")

print('-'.join(items))

print("\n")
