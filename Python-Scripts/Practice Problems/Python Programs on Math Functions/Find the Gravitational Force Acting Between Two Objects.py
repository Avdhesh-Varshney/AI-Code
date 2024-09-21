# Python Program to Find the Gravitational Force Acting Between Two Objects

# Setting value as constant for whole program
G = 6.67430*pow(10,-11)

# Taking input m1, m2, and r
m1 = float(input("\nEnter The Mass of 1st Object (in kg): "))
m2 = float(input("Enter The Mass of 2nd Object (in kg): "))
r = float(input("\nEnter The Displacement Between Them (in meters): "))

# Applying Formula
Force = (G*m1*m2)/(r*r)

# Printing Statement
print(f"\nThe Gravitational Force Acting Between These Two Objects is {round(Force,18)} Newton.\n")
