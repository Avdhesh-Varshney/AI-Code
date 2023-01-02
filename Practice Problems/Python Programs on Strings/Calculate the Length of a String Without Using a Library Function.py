# Python Program to Calculate the Length of a String Without Using a Library Function

def string_length(string):
    count = 0
    for character in string:
        count += 1
    return count

# test the function
test_string = input("\nEnter a string: ")

length = string_length(test_string)

print(f"\nLength of String: {length}\n")
