# Importing string to take UpperCase, LowerCase, Digits & Punctuations in the form of string
import string
# Import random to Shuffle all the Characters
import random

# Creating a function to generate the random password
def passGen():
    # Taking all the UpperCase Letters of Alphabet in a String from 'A' to 'Z'
    str1 = string.ascii_uppercase
    # Taking all the LowerCase Letters of Alphabet in a String from 'a' to 'z'
    str2 = string.ascii_lowercase
    # Taking all the Digits in a String from 0 to 9
    str3 = string.digits
    # Taking all the Symbols, special characters & punctuations in a String
    str4 = string.punctuation

    # Taking input from the user about the length of the password
    passLen = int(input("Enter The Length of Password: "))

    # Creating a empty List
    L = []
    # Inserting all the elements of the strings str1, str2, str3, str4 in the list L
    L.extend(list(str1))
    L.extend(list(str2))
    L.extend(list(str3))
    L.extend(list(str4))

    # After inserting all the elements of the strings in the list
    # Shuffle all the elements so that Password will pick random elements
    random.shuffle(L)
    
    # Password will take random characters of the required length as user wants i.e., passLen
    Password = ("".join(L[0:passLen]))
    
    # Finally, printing the random generated Password
    print(Password)

# Calling the passGen() function
passGen()
