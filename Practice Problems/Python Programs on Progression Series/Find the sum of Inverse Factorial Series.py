# Python Program to Find the Sum of the Series 1/1!+1/2!+1/3!+…1/N!

N = int(input("\nEnter The Limit of The Series: "))
s = 0

print("\nSum of the Series: \n1/1!",end=" + ")

for j in range(2, N):
    print(f"1/{j}!",end=" + ")

print(f"1/{N}!",end=" = ")

for i in range(1,N+1):
    f = 1
    for j in range(1,i+1):
        f = f*j
    s = s + (1/f)

print(f"{s}\n")
