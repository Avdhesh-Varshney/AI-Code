import random

lottery_numbers = []

for i in range(0, 6):
    number = random.randint(1, 50)
    while number in lottery_numbers:
        number = random.randint(1, 50)
    
    lottery_numbers.append(number)

lottery_numbers.sort()

print(lottery_numbers)
