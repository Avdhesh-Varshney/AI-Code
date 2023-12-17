# Python Program to Read a List of Words and Return the Length of the Longest One

words = ['light','split','yellow','present','past','future','independent','dependent','1234687954']

l = len(words[0])

for w in words:
    x = len(w)
    if(x>l):
        l = x

print(f"\nLength of the Longest word: {l}\n")
