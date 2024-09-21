from turtle import *
from time import sleep
speed(80)
color('red')
bgcolor('black')
b = 200
while b > 0:
    left(b)
    forward(b*3)
    b = b-1
sleep(10)
