# Python Program to Find All Numbers which are Odd and Palindromes Between a Range of Numbers without using Recursion

r = int(input("\nEnter the range: "))

print("\nRequired Numbers in the given range: ",end="")

for i in range(1,r+1):
    if(i%2 != 0):
        if(i<10):
            print(i,end="  ")
        else:
            s = str(i)
            c = 0
            for j in range(int(len(s)/2)):
                if(s[j]==s[-1-j]):
                    c += 1
            if(c == int(len(s)/2)):
                print(i,end="  ")

print("\n")
