def count_paths(grid):
    m, n = len(grid), len(grid[0])
    
    # Memoization table
    memo = [[-1] * n for _ in range(m)]
    
    # Helper function to calculate paths using DP with memoization
    def dp(x, y):
        # If out of bounds or if the cell is a hole, return 0 paths
        if x < 0 or y < 0 or grid[x][y] == 1:
            return 0
        # If we are at the start, return 1 path
        if x == 0 and y == 0:
            return 1
        # If already computed, return the cached result
        if memo[x][y] != -1:
            return memo[x][y]
        
        # Calculate the number of paths from the top and left cells
        paths_from_top = dp(x - 1, y)
        paths_from_left = dp(x, y - 1)
        
        # Store the result in the memo table and return
        memo[x][y] = paths_from_top + paths_from_left
        return memo[x][y]
    
    # Start the recursion from the bottom-right cell
    return dp(m - 1, n - 1)

# Example usage:
# 0 represents an open cell and 1 represents a hole
grid = [
    [0, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 1, 0]
]
print("Number of paths:", count_paths(grid))
