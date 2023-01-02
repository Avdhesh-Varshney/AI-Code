# Python Program to Add a Key-Value Pair to the Dictionary

# create an empty dictionary
my_dict = {"1": "King", "2": "Queen", "3": "Joker"}

print(my_dict)

key = input("Enter key: ")
value = input("Enter Value: ")

# add a key-value pair
my_dict.update({key : value})

print(my_dict)
