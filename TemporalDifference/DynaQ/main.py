import random
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # tqdm是显示循环进度条的库

from TemporalDifference import Environment
from TemporalDifference.DynaQ.dyna_q import DynaQ


def DynaQ_CliffWalking(n_planning):
    ncol = 12
    nrow = 4
    env = Environment.env_builder(ncol=ncol, nrow=nrow)
    epsilon = 0.01
    alpha = 0.1
    gamma = 0.9
    agent = DynaQ(ncol, nrow, epsilon, alpha, gamma, n_planning)
    num_episodes = 300  # 智能体在环境中运行多少条序列

    return_list = []  # 记录每一条序列的回报
    for i in range(10):  # 显示10个进度条
        # tqdm的进度条功能
        with tqdm(total=int(num_episodes / 10),
                  desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):  # 每个进度条的序列数
                episode_return = 0
                # 初始化
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)   # 选择动作
                    next_state, reward, done = env.step(action) # 环境反馈
                    episode_return += reward  # 这里回报的计算不进行折扣因子衰减
                    agent.update(state, action, reward, next_state) # 更新
                    state = next_state
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    return return_list

def main():
    np.random.seed(0)
    random.seed(0)
    n_planning_list = [0, 2, 20]
    for n_planning in n_planning_list:
        print('Q-planning步数为：%d' % n_planning)
        time.sleep(0.5)
        return_list = DynaQ_CliffWalking(n_planning)
        episodes_list = list(range(len(return_list)))
        plt.plot(episodes_list,
                 return_list,
                 label=str(n_planning) + ' planning steps')
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Dyna-Q on {}'.format('Cliff Walking'))
    plt.show()

if __name__ == "__main__":
    main()