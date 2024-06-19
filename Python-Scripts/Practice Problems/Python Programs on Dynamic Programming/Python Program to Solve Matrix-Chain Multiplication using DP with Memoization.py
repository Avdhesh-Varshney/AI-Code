# Question link: https://www.geeksforgeeks.org/problems/matrix-chain-multiplication0303/1


def matrix_chain_order(arr):
    n = len(arr) - 1 # Number of matrix is 1 less than lenth of given matrix
    memo = [[None] * n for _ in range(n)] 
    return dp(0, n - 1, memo)

def dp(i, j, memo):
    # Base check
    if i == j:
        return 0
    
    # Memoization check
    if memo[i][j] is not None:
        return memo[i][j]
    
    min_cost = float('inf')

    # Try placing brackets for each 'k' between (i,j)    
    for k in range(i, j):
        cost = dp(i, k) + dp(k + 1, j) + arr[i] * arr[k + 1] * arr[j + 1]
        if cost < min_cost:
            min_cost = cost
    
    # Save the result in the memoization table
    memo[i][j] = min_cost
    return min_cost

# Example usage:
# Matrix dimensions: A1 is 40x20, A2 is 20x30, A3 is 30x10 and A4 is 20x30
arr = [40, 20, 30, 10, 30]

# Optimal value = 26000 [ (A1*(A2*A3))*A4 ]
print("Minimum number of multiplications is", matrix_chain_order(arr))
