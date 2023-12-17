# Python Program to Check if a String is a Pangram or Not

def checkPangram(s):
	List = []

	for i in range(26):
		List.append(False)

	for c in s.lower():
		if not c == " ":
			List[ord(c) -ord('a')] = True

	for ch in List:
		if ch == False:
			return False

	return True

# sentence = "The quick brown fox jumps over the little lazy dog"
sentence = input("Enter The String: ")

if (checkPangram(sentence)):
    print(f"\nEntered String: {sentence} \nThat string is a Pangram.\n")
else:
    print(f"\nEntered String: {sentence} \nThat string is not a Pangram.\n")
