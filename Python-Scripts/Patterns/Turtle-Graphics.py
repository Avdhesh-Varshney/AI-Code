import turtle
turtle.bgcolor("black")
turtle.pensize(1)
turtle.speed(0.2)
colors = ["green", "red", "yellow", "blue", "aqua", "magenta"]

for i in range(20): # 20 times the color will chosen.
    for color in colors:
        turtle.color(color)
        turtle.circle(120)
        turtle.right(10)

turtle.mainloop()

'''
How to make the turtle move in the direction you want it to go.
There are four directions that a turtle can move in:
Forward
Backward
Left
Right
'''
