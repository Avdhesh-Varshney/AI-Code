# Importing Libraries
import time
from random import randint
from playsound import playsound
import threading

# Function will Print all the Emoji's
def HappyNewYear():
   for i in range(1, 85):
      print('')
   space = ''

   for i in range(1, 584):
      count = randint(1, 100)
      while(count > 0):
         space += ' '
         count -= 1
      
      if(i%10 == 0):
         print(space + 'Happy New Year 2023😃')
      elif(i%9 == 0):
         print(space + "🐻‍❄️")
      elif(i%5 == 0):
         print(space + "🐏")
      elif(i%8 == 0):
         print(space + "🌲")
      elif(i%7 == 0):
         print(space + "🎆")
      elif(i%6 == 0):
         print(space + "❤️")
      else:
         print(space + "⛄")
      
      space = ''
      time.sleep(0.2)

# This function will play song
def HappyNewYearSong():
   repeat = 4
   while(repeat > 0):
      # Copy the path of the audio file and paste here
      playsound("D:\Coding\Git\GitHub\PYTHON\Projects\Happy New Year\HappyNewYear.mp3")
      repeat = repeat - 1

# create threads
thread1 = threading.Thread(target=HappyNewYear)
thread2 = threading.Thread(target=HappyNewYearSong)

# start threads
thread1.start()
thread2.start()

# wait for threads to finish
thread1.join()
thread2.join()
