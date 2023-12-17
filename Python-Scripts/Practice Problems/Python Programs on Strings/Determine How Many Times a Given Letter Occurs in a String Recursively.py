# Python Program to Determine How Many Times a Given Letter Occurs in a String Recursively

def checkCountRecursively(string, char):
    if not string:
        return 0

    elif string[0] == char:
        return 1+checkCountRecursively(string[1:], char)

    else:
        return checkCountRecursively(string[1:], char)

s = input("\nEnter some random string = ")
l = input("Enter some random character = ")

print(f"\n{l} character is present {checkCountRecursively(s,l)} times in {s}\n")
