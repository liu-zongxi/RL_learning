import numpy as np
from .solver import Solver


class EpsilonGreedy(Solver):
    """ epsilon贪婪算法,继承Solver类 """
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit) # 父类初始化
        self.epsilon = epsilon
        #初始化拉动所有拉杆的期望奖励估值，默认为1，也就是拉了就有奖励
        # 这影响着决策
        self.estimates = np.array([init_prob] * self.bandit.K)  # 这里的×相当于matlab中的repmat，也就是一排1

    def run_one_step(self):
        # 当前策略选择的k
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.K)  # 随机选择一根拉杆
        else:
            k = np.argmax(self.estimates)  # 选择期望奖励估值最大的拉杆
        # 计算reward
        r = self.bandit.step(k)  # 得到本次动作的奖励
        # 利用reward更新决策
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k