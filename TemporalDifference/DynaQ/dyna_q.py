import numpy as np
import random
import numpy
class DynaQ:
    """ Dyna-Q算法 """
    def __init__(self,
                 ncol,
                 nrow,
                 epsilon,
                 alpha,
                 gamma,
                 n_planning,
                 n_action=4):
        self.Q_table = np.zeros([nrow * ncol, n_action])  # 初始化Q(s,a)表格
        self.n_action = n_action  # 动作个数
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略中的参数

        self.n_planning = n_planning  #执行Q-planning的次数, 对应1次Q-learning
        self.model = dict()  # 环境模型

    def take_action(self, state):  # 选取下一步的操作
        '''epsilon贪婪做出选择的决策方法'''
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def q_learning(self, s0, a0, r, s1):
        '''Qlearning作为Q表的维护方法'''
        td_error = r + self.gamma * self.Q_table[s1].max(
        ) - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error

    def update(self, s0, a0, r, s1):
        self.q_learning(s0, a0, r, s1)  # 更新s0a0的Qtable
        self.model[(s0, a0)] = r, s1  # 将数据添加到模型中，记住这个选择
        # 根据字典的性质，若该数据本身存在于字典中，便不会再一次进行添加
        for _ in range(self.n_planning):  # Q-planning循环
            # 随机选择曾经遇到过的状态动作对
            (s, a), (r, s_) = random.choice(list(self.model.items()))
            # chioce在列表中随机选择一个元素，items返回key和value两个参数
            # 这里根据之前记住的选择再更新一次Qtable，也就是每次都更新N_planning次，这样收敛的更快
            self.q_learning(s, a, r, s_)    # 把之前用到的情况再学一遍，温故而知新