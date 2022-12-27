# Python Program to Find Power of a Number using Recursion

def power_recursively(num, power):
    if power == 1:
        return num
    return num*power_recursively(num, power=power-1)

n = int(input("\nEnter A Number: "))

power = int(input(f"Enter {n} raised to power: "))

ans = power_recursively(n, power)

print(f"\n{n} raised to power {power} is equal to {ans}.\n")
