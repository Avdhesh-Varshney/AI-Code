# Python Program to Find the Largest Number in a List

lis = [1,2,3,4,5,6,7,8,9,10]
l = lis[0]

for i in range(len(lis)):
    if(lis[i]>l):
        l = lis[i]

print(f"\nThe Largest Number in a List: {l}\n")
