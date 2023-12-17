# Python Program to Calculate the Number of Digits and Letters in a String

s = input("\nEnter The String: ")

string = s.split()

d = 0
l = 0
wl = 0

for w in string:
    if(w.isdigit()):
        d += 1
    elif(w.isalpha()):
        wl += 1

for i in range(len(s)):
    for j in range(1,27):
        if(s[i]==chr(64+j)) or (s[i]==chr(96+j)):
            l += 1

print(f"\n{d} Digits and {wl} words containing {l} Letters are Present in The Given String\n")
