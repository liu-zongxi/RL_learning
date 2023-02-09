import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # tqdm是显示循环进度条的库
from TemporalDifference.Environment.cliff_walking_env_TD import CliffWalkingEnvTD
from TemporalDifference.Sarsa.sarsa import Sarsa


def main():
    ncol = 12
    nrow = 4
    env = CliffWalkingEnvTD(ncol, nrow)
    np.random.seed(0)
    epsilon = 0.1
    alpha = 0.1
    gamma = 0.9
    agent = Sarsa(ncol, nrow, epsilon, alpha, gamma)
    num_episodes = 500  # 智能体在环境中运行的序列的数量

    return_list = []  # 记录每一条序列的回报
    for i in range(10):  # 显示10个进度条
        # tqdm的进度条功能
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):  # 每个进度条的序列数
                episode_return = 0
                state = env.reset() # 初始化，回到起点
                action = agent.take_action(state)   # 做出决策
                done = False
                # 循环结束的条件是四元组的最后一个done
                while not done:
                    next_state, reward, done = env.step(action)     # 根据动作获取到下一个状态和奖励
                    next_action = agent.take_action(next_state)     # 根据新的状态对应的Qtable选择一个新的动作
                    episode_return += reward  # 这里回报的计算不进行折扣因子衰减
                    agent.update(state, action, reward, next_state, next_action)    # 更新Qsa进而更新Qtable，注意更新的是当前sa的Qtable
                    state = next_state  # 下一个循环
                    action = next_action    # 这里下一次使用的next_action就是用于更新Q表的action
                return_list.append(episode_return)  # 当前游戏获得的奖励，这是给人看的，不用于训练
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
    plt.title('Sarsa on {}'.format('Cliff Walking'))
    plt.show()
    action_meaning = ['^', 'v', '<', '>']
    print('Sarsa算法最终收敛得到的策略为：')
    print_agent(agent, env, action_meaning, list(range(37, 47)), [47])


def print_agent(agent, env, action_meaning, disaster=[], end=[]):
    for i in range(env.nrow):
        for j in range(env.ncol):
            if (i * env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * env.ncol + j) in end:
                print('EEEE', end=' ')
            else:
                a = agent.best_action(i * env.ncol + j)
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()


if __name__ == "__main__":
    main()