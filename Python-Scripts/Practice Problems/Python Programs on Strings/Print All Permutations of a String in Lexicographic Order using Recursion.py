# Python Program to Print All Permutations of a String in Lexicographic Order using Recursion

# Importing Factorial function from math Library
from math import factorial

def lexicographical_permutations(str):
	for p in range(factorial(len(str))):		
		print(''.join(str), end=', ')
		i = len(str) - 1
		while i > 0 and str[i-1] > str[i]:	
			i -= 1
		str[i:] = reversed(str[i:])
		if i > 0:	
			q = i
			while str[i-1] > str[q]:
				q += 1
			temp = str[i-1]
			str[i-1]= str[q]
			str[q]= temp

s = input("\nEnter The String: ")

print("\nAll The Combinations are Given below: \n")

s = list(s)
s.sort()

lexicographical_permutations(s)

print("\n")
