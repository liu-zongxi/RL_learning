class CliffWalkingEnv:
    """ 悬崖漫步环境"""
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol  # 定义网格世界的列
        self.nrow = nrow  # 定义网格世界的行
        # 转移矩阵P[state][action] = [(p, next_state, reward, done)]包含下一个状态和奖励
        # 每一个sa对应着转移到下一个状态的概率和他的奖励，以及下一个状态是不是游戏结束了
        self.P = self.createP() # 创建状态转移矩阵

    def createP(self):
        # 初始化
        # 生成了4*（row*col）的空列表,对应着4种动作*（row*col）状态
        # 列表中放一个四维元组，对应状态转移概率，状态，奖励，done
        P = [[[] for j in range(4)] for i in range(self.nrow * self.ncol)]
        # 4种动作, change[0]:上,change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        # 定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]] # 动作的坐标变化
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    # 这里就是每一个状态动作对sa

                    # 情况1：位置在悬崖或者目标状态,因为无法继续交互,任何动作奖励都为0
                    if i == self.nrow - 1 and j > 0:
                        # 悬崖处理论上是无法到达的， 所以让他自成死循环，即下一个动作是在原地且没有奖励
                        P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0,
                                                    True)]
                        continue
                    # 情况2：其他位置
                    # 走下一步，遇到边缘对着边缘走等于没走，其他则根据策略来
                    next_x = min(self.ncol - 1, max(0, j + change[a][0]))
                    next_y = min(self.nrow - 1, max(0, i + change[a][1]))
                    # 更新状态，扣一分
                    next_state = next_y * self.ncol + next_x
                    reward = -1
                    # False代表还没结束
                    done = False
                    # 情况3：下一个位置在悬崖或者终点
                    if next_y == self.nrow - 1 and next_x > 0:
                        # True代表游戏结束
                        done = True
                        if next_x != self.ncol - 1:  # 下一个位置在悬崖
                            # 掉悬崖扣100分
                            reward = -100
                    # 更新
                    P[i * self.ncol + j][a] = [(1, next_state, reward, done)]
        return P