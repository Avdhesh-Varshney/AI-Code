# Python Program to Count the Occurrences of Each Word in a Given String Sentence
s = input("\nEnter The String: ")
s.lower()
for i in range(1,27):
    countx = 0
    for j in range(len(s)):
        if(chr(96+i)==s[j]):
            countx += 1
    if(countx > 0):
        print(f"{chr(96+i)} is found {countx} times")
print("\nThanks for using this program.\n")
