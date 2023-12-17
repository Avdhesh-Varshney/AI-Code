# Python Program to Calculate the Average of Numbers in a Given List

l = [1,2,3,4,5,6,7,8,9,10]

sum = 0

for i in range(len(l)):
    sum += l[i]

print(f"\nAverage of All Numbers in a Given List: {float(sum/(len(l)))}\n")
