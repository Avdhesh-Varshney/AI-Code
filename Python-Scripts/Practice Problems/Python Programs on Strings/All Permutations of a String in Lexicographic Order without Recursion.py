# Python Program to Print All Permutations of a String in Lexicographic Order without Recursion

from itertools import permutations

def lexicographical_permutation(str):
	perm = sorted(''.join(chars) for chars in permutations(str))
	for x in perm:
		print(x)

str =input("\nEnter The String: ")

lexicographical_permutation(str)

print("\n")
