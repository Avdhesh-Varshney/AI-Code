# Python Program to Read a Number n And Print the Series “1+2+…..+n= “

n = int(input("\nEnter The Value of n: "))
s = 0

print("\nSeries: ")

for i in range(1,n):
    print(i,end=" + ")
    s = s + i

print(f"{n} = {s+n}\n")
