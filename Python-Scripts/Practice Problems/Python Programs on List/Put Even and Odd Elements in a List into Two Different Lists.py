# Python Program to Put Even and Odd Elements in a List into Two Different Lists

lis = [1,2,3,4,5,6,7,8,9,10]

lis_even = []
lis_odd = []

for i in range(len(lis)):
    if(lis[i]%2 == 0):
        lis_even.append(lis[i])
    else:
        lis_odd.append(lis[i])

print(f"\nRequired Lists are: \n {lis_even} \n {lis_odd} \n")
