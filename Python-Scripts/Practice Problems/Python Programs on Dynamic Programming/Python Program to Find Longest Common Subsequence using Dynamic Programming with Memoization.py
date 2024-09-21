# Question Link: https://atcoder.jp/contests/dp/tasks/dp_f

s = input()
t = input()

dp = [[-1] * (len(t) + 1) for _ in range(len(s) + 1)]

def func(i1, i2):
    # Base check
    if i1 == 0 or i2 == 0:
        return 0

    # Memoization check
    if dp[i1][i2] != -1:
        return dp[i1][i2]

    if s[i1 - 1] == t[i2 - 1]:  # Take the character
        dp[i1][i2] = 1 + func(i1 - 1, i2 - 1)
        return dp[i1][i2]
    else:  # Remove one character from s and t successively and compare
        dp[i1][i2] = max(func(i1, i2 - 1), func(i1 - 1, i2))
        return dp[i1][i2]

lcs = func(len(s), len(t)) # Length of longest common subsequence

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
