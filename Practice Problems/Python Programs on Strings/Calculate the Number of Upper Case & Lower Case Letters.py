# Python Program to Calculate the Number of Upper Case & Lower Case Letters in a String

s = input("\nEnter The String: ")

uc = 0
lc = 0

for i in range(len(s)):

    for j in range(1,27):
        if(s[i]==chr(64+j)):
            uc += 1

    for k in range(1,27):
        if(s[i]==chr(96+k)):
            lc += 1

print(f"\n{uc} Upper Case and {lc} Lower Case Letters are Present in The Given String\n")
