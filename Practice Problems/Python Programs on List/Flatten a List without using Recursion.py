# Python Program to Flatten a List without using Recursion

l = [1, 2, [3, 4, [5, 6], 7],[[[8, 9], 10]],[11, [12, 13]]]

def flatten(input_list):
	result = []
	stack = [input_list]
	while stack:
		current = stack.pop(-1)
		if isinstance(current, list):
			stack.extend(current)
		else:
			result.append(current)
	result.reverse()
	return result

ans = flatten(l)

print(ans)

# By using Direct Library Functions

# from iteration_utilities import deepflatten

# l = [1, 2, [3, 4, [5, 6], 7],
# 	[[[8, 9], 10]], [11, [12, 13]]]

# ans = list(deepflatten(l))

# print(ans)
