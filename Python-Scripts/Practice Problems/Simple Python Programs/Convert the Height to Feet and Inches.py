# Python Program to Read Height in Centimeters and then 
# Convert the Height to Feet and Inches

h = int(input("Enter The Height in cms: "))

f = h*(0.0328084)
i = h*(0.39370079)

print(f"\nHeight in Feet= {round(f,3)} and in Inches= {round(i,3)}.\n")
