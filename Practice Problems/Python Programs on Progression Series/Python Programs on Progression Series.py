# Python Programs on Progression Series

# Function Take input and give Arithmetic Progression series.
def AP():
    a = float(input("\nEnter The First Term: "))
    d = float(input("Enter The Common Difference: "))
    n = float(input("and Number of terms: "))
    s = (n/2)*((2*a)+(n-1)*d)
    l = (a+(n-1)*d)
    print("\nTerms are:",end=" ")
    for i in range(int(n)):
        print(a+(i*d),end="  ")
    print(f"\n\nLast Term of A.P is {l} and\nSum of all the terms of A.P is {s}.\n")

# Function Take input and give Geometric Progression series.
def GP():
    a = float(input("\nEnter The First Term: "))
    r = float(input("Enter The Common Ratio: "))
    n = float(input("and Number of terms: "))
    if(r>1):
        l = a*pow(r,(n-1))
        s = (a*(pow(r,n)-1))/(r-1)
        print("\nTerms are:",end=" ")
        for i in range(1,int(n+1)):
            print(a*pow(r,(i-1)),end="  ")
        print(f"\n\nLast Term of G.P is {l} and\nSum of all the terms of G.P is {s}.\n")
    elif(r==0):
        s = n*a
        print("\nTerms are:",end=" ")
        for i in range(1,int(n+1)):
            print(a,end="  ")
        print(f"\n\nLast Term of G.P is {a} and\nSum of all the terms of G.P is {s}.\n")
    else:
        l = a*pow(r,(n-1))
        s = (a*(1-pow(r,n)))/(1-r)
        print("\nTerms are:",end=" ")
        for i in range(1,int(n+1)):
            print(a*pow(r,(i-1)),end="  ")
        print(f"\n\nLast Term of G.P is {l} and\nSum of all the terms of G.P is {s}.\n")

# Function Take input and give Harmonic Progression series.
def HP():
    a = float(input("\nEnter The First Term: "))
    d = float(input("Enter The Common Difference: "))
    n = float(input("and Number of terms: "))
    s = 1/((n/2)*((2*a)+(n-1)*d))
    l = 1/(a+(n-1)*d)
    print("\nTerms are:",end=" ")
    for i in range(int(n)):
        print(1/(a+(i*d)),end="  ")
    print(f"\nLast Term of H.P is {l} and\nSum of all the terms of H.P is {s}.\n")

# Printing Statement
print('''\n################# Welcome To The Progression Series Program #################\n
Choose '1' for A.P
Choose '2' for G.P
Choose '3' for H.P''')

# Taking Choice from the user
ch = int(input("\nEnter Your choice: "))

if(ch == 1):
    AP()

elif(ch == 2):
    GP()

elif(ch == 3):
    HP()

else:
    print("\nSorry, You are entered Wrong Choice.\n")
