# Python Program to Clear the Rightmost Set Bit of a Number

# Function will return rightmost bit.
def clear_rightmost_set_bit(n):
    # Clear rightmost set bit of n and return it.
    return n & (n - 1)

# Taking Input
n = int(input('Enter a number: '))

# Calculating
ans = clear_rightmost_set_bit(n)

# Printing Statement
print('n with its rightmost set bit cleared equals:', ans)
