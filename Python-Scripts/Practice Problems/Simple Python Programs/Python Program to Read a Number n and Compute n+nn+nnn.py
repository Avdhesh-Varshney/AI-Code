# Python Program to Read a Number n and Compute n+nn+nnn

n = int(input("\nEnter A Number: "))

sum = (n+((n*10)+n)+((n*100)+(n*10)+n))

print(f"\nn+nn+nnn = {n}+{n}{n}+{n}{n}{n} = {sum}.\n")
