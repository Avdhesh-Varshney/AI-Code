# Python Program to Form a Dictionary from an Object of a Class

class base_class(object):
   def __init__(self):
      self.A = 32
      self.B = 60

my_instance = base_class()

print(f"\nAn instance of the class has been created: {my_instance.__dict__}\n")
