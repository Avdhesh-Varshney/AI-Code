# Python Program to Accept Three Digits and Print all Possible Combinations from the Digits

print("\nEnter The Three Digits: \n")

H1 = int(input("1st Digit: "))
H2 = int(input("2nd Digit: "))
H3 = int(input("3rd Digit: "))

print(f"\nAll The Combinations are: \n1. {H1}{H1}{H1}")
print(f"2. {H2}{H2}{H2}\n3. {H3}{H3}{H3}\n4. {H1}{H2}{H3}")
print(f"5. {H1}{H3}{H2}\n6. {H2}{H3}{H1}\n7. {H2}{H1}{H3}")
print(f"8. {H3}{H1}{H2}\n9. {H3}{H2}{H1}\n")
