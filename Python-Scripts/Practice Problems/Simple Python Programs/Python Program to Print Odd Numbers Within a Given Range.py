# Python Program to Print Odd Numbers Within a Given Range

num = int(input("Enter the range: "))

print("\nOdd Numbers are: ",end="")

for i in range(0,num):
    if(i%2==0):
        print(i+1,end=", ")

print("\n")
