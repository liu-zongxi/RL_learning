import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # tqdm是显示循环进度条的库
from TemporalDifference.NstepSarsa.nstep_sarsa import NstepSarsa
from TemporalDifference import Environment


def main():
    ncol = 12
    nrow = 4
    env = Environment.env_builder(ncol=ncol, nrow=nrow)
    np.random.seed(0)
    n_step = 5  # 5步Sarsa算法
    alpha = 0.1
    epsilon = 0.1
    gamma = 0.9
    agent = NstepSarsa(n_step, ncol, nrow, epsilon, alpha, gamma)
    num_episodes = 500  # 智能体在环境中运行的序列的数量

    return_list = []  # 记录每一条序列的回报
    for i in range(10):  # 显示10个进度条
        #tqdm的进度条功能
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):  # 每个进度条的序列数
                episode_return = 0
                state = env.reset() # 初始化
                action = agent.take_action(state)   # 利用epsilon贪婪选择动作
                done = False
                while not done:
                    next_state, reward, done = env.step(action) # 获得sa确定后的下一个状态和奖励
                    next_action = agent.take_action(next_state) # 选择新的动作
                    episode_return += reward  # 这里回报的计算不进行折扣因子衰减
                    agent.update(state, action, reward, next_state, next_action,
                                 done)      # 更新策略，实际上更新的是Nstep前的sa的Q_table
                    state = next_state
                    action = next_action
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('5-step Sarsa on {}'.format('Cliff Walking'))
    plt.show()


if __name__ == "__main__":
    main()