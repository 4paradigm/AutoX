class Solution(object):
    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        res = []
        for i in range(m):
            res.append([0] * n)

        for i in range(m):
            if obstacleGrid[i][0] == 0:
                res[i][0] = 1
            else:
                break
        for j in range(n):
            if obstacleGrid[0][j] == 0:
                res[0][j] = 1
            else:
                break
        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j] == 1:
                    res[i][j] = 0
                else:
                    res[i][j] = res[i - 1][j] + res[i][j - 1]
        return res[-1][-1]

obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
z = Solution()
res = z.uniquePathsWithObstacles(obstacleGrid)
print(res)
