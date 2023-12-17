# Python Program to Find the Intersection of Two Lists

def intersection_list(list1, list2):  
   return set(list1).intersection(list2)  

lis1 = [1,4,6,7,3,6,5,2,7,56]
lis2 = [4,6,2,6,8,34,75,23,76,234,342]

print(f'''\nIntersection of First List: {lis1}\n
and Second List: {lis2} \n\nis: {intersection_list(lis1,lis2)}\n''')
