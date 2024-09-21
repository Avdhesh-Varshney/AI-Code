# Question Link: https://www.geeksforgeeks.org/problems/longest-common-substring1452/1

from typing import List

def longestCommonSubstr(S1: str, S2: str, n: int, m: int) -> int:
    dp: List[List[List[int]]] = [[[-1 for _ in range(2)]] * (m + 1) for _ in range(n + 1)]
    ans = func(S1, S2, 1, 1, dp)
    return max(ans)

def func(s1: str, s2: str, i: int, j: int, dp: List[List[List[int]]]) -> int:
    if i > len(s1) or j > len(s2):
        return (0, 0)

    if dp[i][j][1] != -1:
        return dp[i][j]

    a1 = max(func(s1, s2, i, j + 1, dp))
    a1 = max(a1, max(func(s1, s2, i + 1, j, dp)))
    a2 = [0,-1]
    if s1[i - 1] == s2[j - 1]:
        a2 = func(s1, s2, i + 1, j + 1, dp)

    res = [max(a1, a2[0]), a2[1] + 1]
    dp[i][j] = res
    return dp[i][j]


if __name__ == "__main__":
    print(longestCommonSubstr("abacd", "acaba", 5, 5))
