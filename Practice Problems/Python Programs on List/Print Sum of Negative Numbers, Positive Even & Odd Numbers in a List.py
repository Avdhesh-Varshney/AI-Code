# Python Program to Print Sum of Negative Numbers, Positive Even & Odd Numbers in a List

lis = [-1,-2,-3,-5,-8,-9,-5,6,-45,45,25,52,-9]

sn = 0
spe = 0
spo = 0

for i in range(len(lis)):
    if(lis[i]>0):
        if(lis[i]%2==0):
            spe += lis[i]
        else:
            spo += lis[i]
    else:
        sn += lis[i]

print(f"\nSum of Negative Numbers: {sn}")
print(f"Sum of Positive Even Numbers: {spe}")
print(f"Sum of Positive Odd Numbers: {spo}\n")
