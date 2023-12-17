# Python Program to Find the LCM of Two Numbers

n1 = int(input("\nEnter 1st value: "))
n2 = int(input("and 2nd value: "))

for i in range(1, (n1*n2)+1):
    if(i%n1 == 0) and (i%n2 == 0):
        print(f"\nLCM of {n1} and {n2}: {i}\n")
        break
