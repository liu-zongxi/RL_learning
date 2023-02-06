import numpy as np
import argparse


def config():
    S = ["s1", "s2", "s3", "s4", "s5"]  # 状态集合
    A = ["保持s1", "前往s1", "前往s2", "前往s3", "前往s4", "前往s5", "概率前往"]  # 动作集合
    # 状态转移函数
    P = {
        "s1-保持s1-s1": 1.0,
        "s1-前往s2-s2": 1.0,
        "s2-前往s1-s1": 1.0,
        "s2-前往s3-s3": 1.0,
        "s3-前往s4-s4": 1.0,
        "s3-前往s5-s5": 1.0,
        "s4-前往s5-s5": 1.0,
        "s4-概率前往-s2": 0.2,
        "s4-概率前往-s3": 0.4,
        "s4-概率前往-s4": 0.4,
    }
    # 奖励函数
    R = {
        "s1-保持s1": -1,
        "s1-前往s2": 0,
        "s2-前往s1": -1,
        "s2-前往s3": -2,
        "s3-前往s4": -2,
        "s3-前往s5": 0,
        "s4-前往s5": 10,
        "s4-概率前往": 1,
    }
    gamma = 0.5  # 折扣因子
    MDP = (S, A, P, R, gamma)

    # 策略1,随机策略
    Pi_1 = {
        "s1-保持s1": 0.5,
        "s1-前往s2": 0.5,
        "s2-前往s1": 0.5,
        "s2-前往s3": 0.5,
        "s3-前往s4": 0.5,
        "s3-前往s5": 0.5,
        "s4-前往s5": 0.5,
        "s4-概率前往": 0.5,
    }
    # 策略2
    Pi_2 = {
        "s1-保持s1": 0.6,
        "s1-前往s2": 0.4,
        "s2-前往s1": 0.3,
        "s2-前往s3": 0.7,
        "s3-前往s4": 0.5,
        "s3-前往s5": 0.5,
        "s4-前往s5": 0.1,
        "s4-概率前往": 0.9,
    }
    Pis = (Pi_1, Pi_2)
    return MDP, Pis


# 把输入的两个字符串通过“-”连接,便于使用上述定义的P、R变量
def join(str1, str2):
    return str1 + '-' + str2


def main_MC(Pi_index):
    np.random.seed(0)
    MDP, Pis = config()
    Pi = Pis[Pi_index]
    # print(Pi)
    timestep_max = 20
    # 采样1000次,可以自行修改
    episodes = sample(MDP, Pi, timestep_max, 1000)
    gamma = 0.5
    V = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}
    N = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}
    MC(episodes, V, N, gamma)
    print("使用蒙特卡洛方法计算MDP的状态价值为\n", V)


def main_occupancy():
    np.random.seed(0)
    MDP, Pis = config()
    Pi_1 = Pis[0]
    Pi_2 = Pis[1]
    gamma = 0.5
    timestep_max = 1000

    episodes_1 = sample(MDP, Pi_1, timestep_max, 1000)
    episodes_2 = sample(MDP, Pi_2, timestep_max, 1000)
    rho_1 = occupancy(episodes_1, "s4", "概率前往", timestep_max, gamma)
    rho_2 = occupancy(episodes_2, "s4", "概率前往", timestep_max, gamma)
    print(rho_1, rho_2)
def compute(P, rewards, gamma, states_num):
    ''' 利用贝尔曼方程的矩阵形式计算解析解,states_num是MRP的状态数 '''
    rewards = np.array(rewards).reshape((-1, 1))  #将rewards写成列向量形式
    value = np.dot(np.linalg.inv(np.eye(states_num, states_num) - gamma * P),
                   rewards)
    return value


def main_V_analytical_solution():
    gamma = 0.5
    # 转化后的MRP的状态转移矩阵
    P_from_mdp_to_mrp = [
        [0.5, 0.5, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.5, 0.5],
        [0.0, 0.1, 0.2, 0.2, 0.5],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ]
    P_from_mdp_to_mrp = np.array(P_from_mdp_to_mrp)
    R_from_mdp_to_mrp = [-0.5, -1.5, -1.0, 5.5, 0]

    V = compute(P_from_mdp_to_mrp, R_from_mdp_to_mrp, gamma, 5)
    print("MDP中每个状态价值分别为\n", V)


def sample(MDP, Pi, timestep_max, number):
    ''' 采样函数,策略Pi,限制最长时间步timestep_max,总共采样序列数number
        本质上采样了number个路径'''
    S, A, P, R, gamma = MDP
    episodes = []
    for _ in range(number):
        episode = []
        timestep = 0
        s = S[np.random.randint(4)]  # 随机选择一个除s5以外的状态s作为起点，因为S5是终点
        # 当前状态为终止状态或者时间步太长时,一次采样结束
        while s != "s5" and timestep <= timestep_max:
            timestep += 1
            # 以下是一种很好的对已知分布获得采样的方法
            # 本质是对分布CDF进行采样，来获得采样值，值得学习
            rand, temp = np.random.rand(), 0
            # 在状态s下根据策略选择动作
            for a_opt in A:
                temp += Pi.get(join(s, a_opt), 0)   # 得到CDF,get是字典，0是在没有对应值时返回0
                # 对CDF进行采样
                if temp > rand:
                    a = a_opt
                    r = R.get(join(s, a), 0)
                    break
            rand, temp = np.random.rand(), 0
            # 根据状态转移概率得到下一个状态s_next
            for s_opt in S:
                # 对于伯努利分布，这个方法也是同样适用的
                temp += P.get(join(join(s, a), s_opt), 0)
                if temp > rand:
                    s_next = s_opt
                    break
            episode.append((s, a, r, s_next))  # 把（s,a,r,s_next）元组放入序列中
            s = s_next  # s_next变成当前状态,开始接下来的循环
        episodes.append(episode)
    return episodes


# 对所有采样序列计算所有状态的价值
def MC(episodes, V, N, gamma):
    for episode in episodes:
        G = 0
        for i in range(len(episode) - 1, -1, -1):  #一个序列从后往前计算，range是一个左闭右开区间
            # 从后往前计算的原因是为了处理遗传因子
            (s, a, r, s_next) = episode[i]
            G = r + gamma * G
            # 增量式的更新期望
            N[s] = N[s] + 1
            V[s] = V[s] + (G - V[s]) / N[s]


def occupancy(episodes, s, a, timestep_max, gamma):
    ''' 计算状态动作对（s,a）出现的频率,以此来估算策略的占用度量 '''
    rho = 0
    total_times = np.zeros(timestep_max)  # 记录每个时间步t各被经历过几次
    occur_times = np.zeros(timestep_max)  # 记录(s_t,a_t)=(s,a)的次数
    for episode in episodes:
        # 这里只是做简单的遍历，因此不需要倒叙
        for i in range(len(episode)):
            (s_opt, a_opt, r, s_next) = episode[i]
            total_times[i] += 1
            if s == s_opt and a == a_opt:
                occur_times[i] += 1
    # 这里的reversed没有意义
    # for i in reversed(range(timestep_max)):
    for i in range(timestep_max):
        if total_times[i]:
            rho += gamma**i * occur_times[i] / total_times[i]
    return (1 - gamma) * rho


if __name__ == '__main__':
    # main_V_analytical_solution()
    # main_MC(0)
    main_occupancy()