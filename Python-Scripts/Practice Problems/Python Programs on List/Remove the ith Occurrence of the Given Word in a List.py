# Python Program to Remove the ith Occurrence of the Given Word in a List

def removeWord(list, word, k):
    n = 0
    for i in range(len(list)):
        if(list[i] == word):
            n += 1
            if(n == k):
                del(list[i])
                return True
    return False

list = ['you','light','split','yellow','hello','you','present','past','future','independent','hello']

if(removeWord(list, input("\nEnter a word: "), int(input("Enter k: ")))):
    print(f"\nThe list is updated : {list}\n")
else:
    print("\nThe given word is not found\n")
