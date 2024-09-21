# Question link: https://atcoder.jp/contests/dp/tasks/dp_d

from typing import List

n, w = map(int, input().split())
weight = [0] * n
price = [0] * n

for i in range(n):
    weight[i], price[i] = map(int, input().split())

dp = [[-1] * (w + 1) for _ in range(n + 1)]

# Base case
for j in range(w + 1):
    dp[0][j] = 0

for i in range(1, n + 1):
    for j in range(w + 1):

        # Don't Take current item
        dp[i][j] = dp[i - 1][j]

        # Take current item
        if j - weight[i - 1] >= 0:
            dp[i][j] = max(dp[i][j], dp[i - 1][j - weight[i - 1]] + price[i - 1])

print(dp[n][w])

