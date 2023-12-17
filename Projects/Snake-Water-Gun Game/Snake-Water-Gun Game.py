''' <======================================== Project: Snake, Water, Gun Game ========================================>

We all have played snake, water, gun game in our childhood. If you haven't, google the rules of the game and play it.'''

# Snake Water Gun ============ OR =========== Rock Paper Scissors

# Import Library
import random

# This function check the rules and return Either None or True or False
def gameWin(comp, you):
    # If two values are equal, declare a tie!
    if comp == you:
        return None

    # Check for all possibilities when computer chose Snake(s)
    elif comp == 's':
        if you == 'w':
            return False
        elif you == 'g':
            return True

    # Check for all possibilities when computer chose Water(w)
    elif comp == 'w':
        if you == 'g':
            return False
        elif you == 's':
            return True

    # Check for all possibilities when computer chose Gun(g)
    elif comp == 'g':
        if you == 's':
            return False
        elif you == 'w':
            return True

# Computer Choose randomly
randNo = random.randint(1, 3)
if randNo == 1:
    comp = 's'
elif randNo == 2:
    comp = 'w'
elif randNo == 3:
    comp = 'g'

print("\nComputer's Turn: Snake(s) Water(w) or Gun(g)? : Done" )

while(True):
    # Taking Input
    print('''\nYour Turn: 
    For Snake, Choose (s) 
    For Water, Choose (w) or 
    For Gun, Choose (g)\n''')

    you = input("It's Your Turn: ")

    # Checking what you have Entered is Valid or Invalid
    if(you == 's') or (you == 'w') or (you == 'g'):
        break
    else:
        print("\nInvalid Input")

# Checking the conditions
a = gameWin(comp, you)

# Check what the computer has choose
if comp == 's':
    print("Computer Chosed Snake")
elif comp == 'w':
    print("Computer Chosed Water")
elif comp == 'g':
    print("Computer Chosed Gun")

# Check what you have choose
if you == 's':
    print("You Chosed Snake")
elif you == 'w':
    print("You Chosed Water")
elif you == 'g':
    print("You Chosed Gun")

# Check whether you get tie, win or lose!
if a == None:
    print("The game is a tie!")
elif a:
    print("You Win!")
else:
    print("You Lose!")
