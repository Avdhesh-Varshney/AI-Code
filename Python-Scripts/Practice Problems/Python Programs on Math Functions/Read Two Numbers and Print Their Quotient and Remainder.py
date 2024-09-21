# Python Program to Read Two Numbers and Print Their Quotient and Remainder

# Taking Divisor and Dividend as input
divisor = float(input("\nEnter The Values of Divisor: "))
dividend = float(input("\nEnter The Values of Dividend: "))

# By dividing we got quotient as integer part
Quotient = int(dividend/divisor)

# Applying modulus then, we got remainder as output
Remainder = int(dividend%divisor)

# Printing Statement
print(f"\nQuotient= {Quotient} and Remainder= {Remainder}.\n")
