import numpy as np
from .solver import Solver


class UCB(Solver):
    """ UCB算法,继承Solver类 """
    def __init__(self, bandit, coef, init_prob=1.0):
        super(UCB, self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.coef = coef    # 不确定度的放缩系数

    def run_one_step(self):
        # 做出决策
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(
            np.log(self.total_count) / (2 * (self.counts + 1)))  # 计算上置信界，设置了概率随次数变化，本身和每个老虎机被拉动的次数有关
        k = np.argmax(ucb)  # 选出上置信界最大的拉杆
        # 给出当前决策的奖励
        r = self.bandit.step(k)
        # 根据reward改变决策
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k


