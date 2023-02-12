import random
import gym
import numpy as np
from tqdm import tqdm
import torch
from DQN.model.dqn import DQN
from DQN.model.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
import rl_utils


def main():
    lr = 2e-3
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 10
    buffer_size = 10000
    minimal_size = 500
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")
    print(device)
    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)
    replay_buffer = ReplayBuffer(buffer_size)   # 初始化一个经验回放池，它能够回放记忆决策和结果
    state_dim = env.observation_space.shape[0]  # 在本场景中为4维为车的位置速度，杆的角度和角速度
    action_dim = env.action_space.n # 2维，向左或是右
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
                target_update, device)  # 初始化一个智能体

    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset() # 初始化状态
                done = False
                while not done:
                    action = agent.take_action(state)   # 选择一个动作
                    next_state, reward, done, _ = env.step(action)  # 与环境交互得到下一个状态和奖励
                    replay_buffer.add(state, action, reward, next_state, done)  # 这次交互结果被记忆下来
                    state = next_state
                    episode_return += reward
                    # 当buffer数据的数量超过一定值后,才进行Q网络训练
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size) # 取出batch_size大小的结果训练
                        # 存入字典,每一个都有batch_size大小
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        # 更新Q网络的参数
                        # 使用当前sa和Q网络计算出Qsa
                        # 使用下一个s和贪婪算法，更新Q_target
                        # 然后计算loss进行反向传播
                        agent.update(transition_dict)
                return_list.append(episode_return)
                # 输出结果
                if (i_episode + 1) % 10 == 0:
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
    plt.title('DQN on {}'.format(env_name))
    plt.show()
    # 滑动取均值，也就是最近9个的结果
    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format(env_name))
    plt.show()


if __name__ == "__main__":
    main()