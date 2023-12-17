# Python Program to Check if a Given Key Exists in a Dictionary or Not

def checkKey(dict, key):
	if key in dict.keys():
		print("\nPresent, ", end =" ")
		print("value =", dict[key])
	else:
		print("Not present")

dict = {'a': 100, 'b':200, 'c':300}

key = 'b'
checkKey(dict, key)
print("\n")

key = 'w'
checkKey(dict, key)
print("\n")
