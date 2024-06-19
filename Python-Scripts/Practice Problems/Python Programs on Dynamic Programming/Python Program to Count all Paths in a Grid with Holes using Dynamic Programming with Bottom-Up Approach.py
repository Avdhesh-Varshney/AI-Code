def count_paths(grid):
    m, n = len(grid), len(grid[0])
    
    # Create a 2D dp array to store the number of paths to each cell
    dp = [[0] * n for _ in range(m)]
    
    # Initialize the starting position if it's not a hole
    if grid[0][0] == 0:
        dp[0][0] = 1
    
    # Fill the dp table
    for i in range(m):
        for j in range(n):
            # If the cell is a hole, skip it
            if grid[i][j] == 1:
                dp[i][j] = 0
            else:
                # Add the number of paths from the top cell if it exists
                if i > 0:
                    dp[i][j] += dp[i - 1][j]
                # Add the number of paths from the left cell if it exists
                if j > 0:
                    dp[i][j] += dp[i][j - 1]
    
    # The answer is the number of paths to the bottom-right cell
    return dp[m - 1][n - 1]

# Example usage:
# 0 represents an open cell and 1 represents a hole
grid = [
    [0, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 1, 0]
]
print("Number of paths:", count_paths(grid))
