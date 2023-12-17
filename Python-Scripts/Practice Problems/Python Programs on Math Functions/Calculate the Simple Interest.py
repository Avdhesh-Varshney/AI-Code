# Python Program to Calculate the Simple Interest

# Taking input as Principal, Rate, Time
p = float(input("\nEnter The Principal Amount: "))
r = float(input("\nEnter The Rate: "))
t = float(input("\nEnter The Time Period: "))

# Applying Formula
si = (p*r*t)/100

# Printing Statement
print(f"\nSimple Interest of Given Values are: {si}\n")
