# Python Program to Find the Sum of Elements in a List Recursively

def sum_arr(arr,size):
   if (size == 0):
     return 0
   else:
     return arr[size-1] + sum_arr(arr,size-1)

a = [1,34,65,34,65,23,78,45,8,23,24,34,8,8]

print("\nThe list is: ",end="")

print(a)
print("\nSum of items in list:",end=" ")

b=sum_arr(a,len(a))
print(f"{b}\n")
