import numpy as np


# 给定一条序列,计算从某个索引（起始状态）开始到序列最后（终止状态）得到的回报
def compute_return(start_index, chain, gamma, rewards):
    G = 0
    for i in reversed(range(start_index, len(chain))):
        G = gamma * G + rewards[chain[i] - 1]
    return G

def compute(P, rewards, gamma, states_num):
    ''' 利用贝尔曼方程的矩阵形式计算解析解,states_num是MRP的状态数 '''
    rewards = np.array(rewards).reshape((-1, 1))  #将rewards写成列向量形式,-1是自动计算
    value = np.dot(np.linalg.inv(np.eye(states_num, states_num) - gamma * P),
                   rewards) # np.dot是矩阵乘法，np.linalg.inv()是逆矩阵，即实现贝尔曼方程的解
    return value




def main():
    np.random.seed(0)
    # 定义状态转移概率矩阵P
    P = [
        [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
        [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
        [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]
    P = np.array(P)

    rewards = [-1, -2, -2, 10, 1, 0]  # 定义奖励函数
    gamma = 0.5  # 定义折扣因子
# # 一个状态序列,s1-s2-s3-s6
#     chain = [1, 2, 3, 6]
#     start_index = 0
#     G = compute_return(start_index, chain, gamma, rewards)
#     print("根据本序列计算得到回报为：%s。" % G)
    V = compute(P, rewards, gamma, 6)
    print("MRP中每个状态价值分别为\n", V)


if __name__ == '__main__':
    main()