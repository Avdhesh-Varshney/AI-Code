# Python Program to Find the Second Largest Number in a List

lis = [1,2,3,4,5,6,7,8,9,10]

l = lis[0]
l2 = l

for i in range(len(lis)):
    if(lis[i]>l):
        l2 = l
        l = lis[i]

print(f"\nSecond Largest Number in a List: {l2}\n")
