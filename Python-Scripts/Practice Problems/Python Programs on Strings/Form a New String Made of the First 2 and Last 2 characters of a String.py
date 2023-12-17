# Python Program to Form a New String Made of the First 2 and Last 2 characters of a String

string = input("\nEnter The String: ")

count = 0

new = string[0:2] + string[len(string)-2:len(string)]

print(f"\nNewly formed string is: {new}\n")
