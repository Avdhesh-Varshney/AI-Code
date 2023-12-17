# Python Program to Find the Cumulative Sum of a List

a = [1,34,65,34,65,23,78,45,8,23,24,34,8,8]
sum = 0

print("\nThe Cumulative Sum of a List: \n")
for i in range(len(a)):
    sum += a[i]
    print(sum,end="  ")

print("\n")
