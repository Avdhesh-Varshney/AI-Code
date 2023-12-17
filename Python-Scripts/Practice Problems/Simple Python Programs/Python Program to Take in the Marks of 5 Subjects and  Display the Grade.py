# Python Program to Take in the Marks of 5 Subjects and 
# Display the Grade

n1 = int(input("\nEnter The Marks of English: "))
n2 = int(input("\nEnter The Marks of Hindi: "))
n3 = int(input("\nEnter The Marks of Maths: "))
n4 = int(input("\nEnter The Marks of Science: "))
n5 = int(input("\nEnter The Marks of S.St: "))

avg = (n1+n2+n3+n4+n5)/5

if(avg>=90):
    print("\nCongratulation! You Got H Grade\n")
elif(avg>=80):
    print("\nCongratulation! You Got A Grade\n")
elif(avg>=70):
    print("\nCongratulation! You Got B Grade\n")
elif(avg>=60):
    print("\nCongratulation! You Got C Grade\n")
elif(avg>=50):
    print("\nCongratulation! You Got D Grade\n")
elif(avg>=33):
    print("\nCongratulation! You Got E Grade\n")
