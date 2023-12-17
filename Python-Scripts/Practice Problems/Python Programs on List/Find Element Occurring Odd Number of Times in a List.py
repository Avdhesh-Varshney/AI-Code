# Python Program to Find Element Occurring Odd Number of Times in a List

a = [1,34,65,34,65,23,78,45,8,23,24,34,8,8]
lis = []

for i in range(len(a)):
    countx = 0
    for j in range(len(a)):
        if(a[i]==a[j]):
            countx += 1
    if(countx%2 != 0):
        lis.append(a[i])

lis = sorted(list(set(lis)))

print(f"\nRequired Numbers: {lis}\n")
