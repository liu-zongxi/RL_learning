import gym
from CliffWalking.value_iteration import ValueIteration
from CliffWalking.policy_iteration import PolicyIteration
def env_generator():
    env = gym.make("FrozenLake-v1")  # 创建环境
    env = env.unwrapped  # 解封装才能访问状态转移矩阵P
    env.render()  # 环境渲染,通常是弹窗显示或打印出可视化的环境
    return env

def env_observation(env):
    holes = set()
    ends = set()
    for s in env.P:
        for a in env.P[s]:
            for s_ in env.P[s][a]:
                if s_[2] == 1.0:  # 获得奖励为1,代表是目标
                    ends.add(s_[1])
                if s_[3] == True:
                    holes.add(s_[1])
    holes = holes - ends
    print("冰洞的索引:", holes)
    print("目标的索引:", ends)

    for a in env.P[14]:  # 查看目标左边一格的状态转移信息
        print(env.P[14][a])

def print_agent(agent, action_meaning, disaster=[], end=[]):
    print("状态价值：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 为了输出美观,保持输出6个字符
            print('%6.6s' % ('%.3f' % agent.v[i * agent.env.ncol + j]), end=' ')
        print()

    print("策略：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 一些特殊的状态,例如悬崖漫步中的悬崖
            if (i * agent.env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * agent.env.ncol + j) in end:  # 目标状态
                print('EEEE', end=' ')
            else:
                a = agent.pi[i * agent.env.ncol + j]
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()

def main():
    env = env_generator()
    env_observation(env)
    action_meaning = ['<', 'v', '>', '^']
    theta = 1e-5
    gamma = 0.9
    agent = ValueIteration(env, theta, gamma)
    agent.value_iteration()
    print_agent(agent, action_meaning, [5, 7, 11, 12], [15])


if __name__ == "__main__":
    main()