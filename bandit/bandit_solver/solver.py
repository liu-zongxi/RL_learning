import numpy as np

class Solver:
    """ 多臂老虎机算法基本框架 """
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)  # 每根拉杆的尝试次数
        self.regret = 0.  # 当前步的累积懊悔
        self.actions = []  # 维护一个列表,记录每一步的动作
        self.regrets = []  # 维护一个列表,记录每一步的累积懊悔

    def update_regret(self, k):
        # 计算累积懊悔并保存,k为本次动作选择的拉杆的编号
        self.regret += self.bandit.best_prob - self.bandit.probs[k]     # 懊悔是最佳选择和当前选择的差
        self.regrets.append(self.regret)    # 记录当前的regret

    def run_one_step(self):
        # 返回当前动作选择哪一根拉杆,由每个具体的策略实现
        # 虚函数
        raise NotImplementedError

    def run(self, num_steps):
        # 运行一定次数,num_steps为总运行次数
        for _ in range(num_steps):
            k = self.run_one_step()     # 一次运行中会更新策略，并获得当前策略的选择结果
            self.counts[k] += 1         # 老虎机被拉的次数+1
            self.actions.append(k)      # 记录决策
            self.update_regret(k)       # 记录regret