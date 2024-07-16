# Question Link: https://www.geeksforgeeks.org/problems/longest-common-substring1452/1

from typing import List

def longestCommonSubstr(S1: str, S2: str, n: int, m: int) -> int:
    dp: List[List[int]] = [[[-1]] * (m + 1) for _ in range(n + 1)]

    for i in range(n+1):
        dp[i][0] = 0
    for i in range(m+1):
        dp[0][i] = 0
    
    for i in range(1,n+1):
        for j in range(1,m+1):
            if S1[i-1] == S2[j-1]:
                dp[i][j] = dp[i-1][j-1]+1
            else:
                dp[i][j]=0

    return max(max(dp,key=max))


if __name__ == "__main__":
    print(longestCommonSubstr("abacd", "acaba", 5, 5))
