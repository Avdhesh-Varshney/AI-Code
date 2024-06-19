n = int(input("Enter number: "))

dp = [-1]*(n+1) # Creation of DP array
dp[0] = 0       # Base case

def fib(n): 
    """
    Calculates the nth Fibonacci number using dynamic programming with memoization.
    Parameters:
        n (int): The index of the Fibonacci number to calculate.
    Returns:
        int: The nth Fibonacci number.
    """
    # Base case
    if n == 0:
        return 0
    if n == 1:
        dp[1] = 1
        return 1
    
    # Memoization check
    if dp[n] != -1:
        return dp[n]
    
    # Recurrence relation
    dp[n] = fib(n-1) + fib(n-2)
    return dp[n]

print(fib(n))