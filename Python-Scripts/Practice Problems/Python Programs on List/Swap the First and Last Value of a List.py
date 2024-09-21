# Python Program to Swap the First and Last Value of a List

a = [1,34,65,34,65,23,78,45,8,23,24,34,8,8]

print(f"\nEntered String: {a}\n")

x = a[0]
a[0] = a[-1]
a[-1] = x

print(f"\nRequired String: {a}\n")
