# https://atcoder.jp/contests/dp/tasks/dp_f

s = input()
t = input()

dp = [[-1] * (len(t) + 1) for _ in range(len(s) + 1)]

# Base check
dp[0] = [0]*(len(t) + 1)
for i in range(len(s)+1):
    dp[i][0]=0

for i1 in range(1,len(s)+1):
    for i2 in range(1,len(t)+1):
        if s[i1 - 1] == t[i2 - 1]:  # Take the character
            dp[i1][i2] = 1 + dp[i1 - 1][i2 - 1]

        else:  # Remove one character from s and t successively and compare
            dp[i1][i2] = max(dp[i1][i2-1],dp[i1-1][i2])

lcs = dp[len(s)][len(t)] # Lenght of longest common subsequence
# print(lcs)

ans = [] # Find the longest common subsequence
i = len(s)
j = len(t)
while (i > 0 and j > 0):
    if (s[i - 1] == t[j - 1]):
        ans.append(s[i - 1])
        i-=1
        j-=1
    elif (dp[i - 1][j] < dp[i][j - 1]):
        j-=1
    else:
        i-=1

print(''.join(ans[::-1]))

