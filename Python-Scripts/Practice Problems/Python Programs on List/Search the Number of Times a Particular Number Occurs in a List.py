# Python Program to Search the Number of Times a Particular Number Occurs in a List

lis = [1,34,65,34,65,23,78,45,8,23,24,34,8,8]

print("\n")

for i in range(len(lis)):
    countx = 0
    for j in range(len(lis)):
        if(lis[i]==lis[j]):
            countx += 1
    print(f"{lis[i]} is present {countx} times")

print("\n")
