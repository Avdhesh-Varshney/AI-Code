# Python Program to Determine all Pythagorean Triplets in the Range

print("\nEnter The Pythagorean Triplets: \n")
a = int(input("a = "))
b = int(input("b = "))
c = int(input("c = "))

if(a>b):
    if(a>c):
        z = a
        if(b>c):
            y = b
            x = c
        else:
            y = c
            x = b

else:
    if(b>c):
        z = b
        if(a>c):
            y = a
            x = c
        else:
            y = c
            x = a

if((x*x)+(y*y)==(z*z)):
    print(f"\n({x}, {y}, {z}) Numbers are Pythagorean Triplets.\n")

else:
    print(f"\n({x}, {y}, {z}) Numbers are not Pythagorean Triplets.\n")

