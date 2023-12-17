# Python Program to Form an Integer that has the Number of Digits at Ten’s Place and the Least Significant Digit of the Entered Integer at One’s Place

n = int(input("\nEnter A Number: "))

if(n>0):
    l = int(len(str(n)))
    s = n%10

else:
    l = int(len(str(-1*n)))
    s = (-1*n)%10

print(f"\nRequired Number is: {(l*10)+s}\n")
