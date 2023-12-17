# Python Program to Print Largest Even and Largest Odd Number in a List

lis = [1,2,3,4,5,6,7,8,9,10]

le = lis[0]
lo = lis[0]

for i in range(len(lis)):
    if(lis[i]%2 == 0)and(lis[i]>le):
        le = lis[i]
    elif(lis[i]%2 != 0)and(lis[i]>lo):
        lo = lis[i]

print(f"\nLargest Even Number in a List: {le}\n")
print(f"Largest Odd Number in a List: {lo}\n")
