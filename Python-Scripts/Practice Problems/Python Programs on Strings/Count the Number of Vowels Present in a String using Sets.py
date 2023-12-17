# Python Program to Count the Number of Vowels Present in a String using Sets

s = input("\nEnter The String: ")

vowel = ("aeiouAEIOU")

coutnx = 0

for a in s:
    if a in vowel:
        coutnx += 1

print(f"\nNumber of Vowels are: {coutnx}\n")
