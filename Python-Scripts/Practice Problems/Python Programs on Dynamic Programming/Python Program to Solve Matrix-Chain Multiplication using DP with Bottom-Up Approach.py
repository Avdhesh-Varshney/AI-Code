# Question link: https://www.geeksforgeeks.org/problems/matrix-chain-multiplication0303/1

def matrix_chain_order(arr):
    # Number of matrices
    n = len(arr) - 1
    
    # Create a 2D array to store the minimum number of multiplications needed to compute the matrix A[i]...A[j]
    dp = [[0] * n for _ in range(n)]
    
    # l is the chain length
    for l in range(2, n + 1):
        for i in range(n - l + 1):
            j = i + l - 1
            dp[i][j] = float('inf')
            for k in range(i, j):
                ans = dp[i][k] + dp[k + 1][j] + arr[i] * arr[k + 1] * arr[j + 1]
                if ans < dp[i][j]:
                    dp[i][j] = ans
    
    return dp[0][n - 1]

# Example usage:
# Matrix dimensions: A1 is 40x20, A2 is 20x30, A3 is 30x10 and A4 is 20x30
arr = [40, 20, 30, 10, 30]

# Optimal value = 26000 [ (A1*(A2*A3))*A4 ]
print("Minimum number of multiplications is", matrix_chain_order(arr))
