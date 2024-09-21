# Question link: https://atcoder.jp/contests/dp/tasks/dp_d

from typing import List

n, w = map(int, input().split())
weight = [0] * n
price = [0] * n

for i in range(n):
    weight[i], price[i] = map(int, input().split())

dp = [[-1] * (w + 1) for _ in range(n + 1)]

def func(weight: List[int], price: List[int], idx: int, cur: int) -> int:
    if idx == 0:
        return 0

    if dp[idx][cur] != -1:
        return dp[idx][cur]

    # Not take
    ans = func(weight, price, idx - 1, cur)

    # Take
    if cur - weight[idx-1] >= 0:
        ans = max(ans, func(weight, price, idx - 1, cur - weight[idx-1]) + price[idx-1])

    dp[idx][cur] = ans
    return ans

print(func(weight, price, n, w))

