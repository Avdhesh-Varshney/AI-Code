# Question Link : https://www.geeksforgeeks.org/problems/rod-cutting0840/1

# Method 1 (Bottom-up Approach)

n = int(input())
price = list(map(int,input().split()))

dp=price.copy() # Initialize DP with base values as the price (as the rod can be divided into atleast i length, 1<=i<n)
dp.insert(0,0) # For base case and to change it to 1-based indexing

for i in range(1,n+1):
    for j in range(i+1): # To get the most optimal way to divide the rod of length of i
        dp[i] = max(dp[i],dp[j]+dp[i-j])

print(dp[n]) # Time complexity : O(n^2)