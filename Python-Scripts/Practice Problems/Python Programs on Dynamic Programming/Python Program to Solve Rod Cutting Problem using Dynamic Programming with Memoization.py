# Question Link : https://www.geeksforgeeks.org/problems/rod-cutting0840/1

# Method 1 (Memoization)
import sys
sys.setrecursionlimit(10**5) # As the maximum recursion depth of Python is 1000, modify if required

n=int(input())
price = list(map(int,input().split()))

dp=[-1]*(n+1) # Creation of DP array
dp[0]=0

def func(prices, n, dp):
    if n <= 0:
        return 0
        
    if dp[n] != -1:
        return dp[n]  # Return already calculated value

    ans = 0
    # Recursively cut the rod and update the maximum profit
    for i in range(1, n + 1):
        ans = max(ans, prices[i-1] + func(prices, n - i,dp))

    dp[n] = ans  # Store the calculated maximum profit
    return dp[n]


print(func(price,n,dp))

