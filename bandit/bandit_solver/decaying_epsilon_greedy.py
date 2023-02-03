import numpy as np
from .solver import Solver


class DecayingEpsilonGreedy(Solver):
    """ epsilon值随时间衰减的epsilon-贪婪算法,继承Solver类 """
    def __init__(self, bandit, init_prob=1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.total_count = 0    # 计数，这和衰减有关

    def run_one_step(self):
        # 做出决策
        self.total_count += 1
        if np.random.random() < 1 / self.total_count:  # epsilon值随时间衰减
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)
        # 获得reward
        r = self.bandit.step(k)
        # 更新策略
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])

        return k


