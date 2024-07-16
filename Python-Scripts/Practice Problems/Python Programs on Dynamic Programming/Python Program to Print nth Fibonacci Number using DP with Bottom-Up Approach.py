n = int(input("Enter number: "))

if n<=1 : 
    print(n)
    exit(1)

dp = [-1]*(n+1) # Creation of DP array
dp[0] = 0       # Base cases
dp[1] = 1

for i in range(2,n+1):
    dp[i] = dp[i-1] +dp[i-2] # Recurrence relation

print(dp[n])