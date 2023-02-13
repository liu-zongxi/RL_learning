import torch
from Reinforce.model.policy_net import PolicyNet

class REINFORCE:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 device):
        self.policy_net = PolicyNet(state_dim, hidden_dim,
                                    action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                          lr=learning_rate)  # 使用Adam优化器
        self.gamma = gamma  # 折扣因子
        self.device = device

    def take_action(self, state):  # 根据动作概率分布随机采样
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.policy_net(state)  # 做前向传播得到决策分布
        action_dist = torch.distributions.Categorical(probs)    # torch自带的根据概率分布进行采样
        action = action_dist.sample()   # 根据决策采样得到一个action
        return action.item()    # tensor中取数

    def update(self, transition_dict):
        # 输入的字典中包含了一个episode的完整记录
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0
        self.optimizer.zero_grad()  # 清空梯度
        for i in reversed(range(len(reward_list))):  # 从最后一步算起
            reward = reward_list[i]
            state = torch.tensor([state_list[i]],
                                 dtype=torch.float).to(self.device) # 长度为state_dim
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action))  # 选择action对应的概率
            G = self.gamma * G + reward # 更新G
            loss = -log_prob * G  # 每一步的损失函数
            loss.backward()  # 反向传播计算梯度，更新梯度
            # 注意这里pytroch的性质，梯度会自动累加
        self.optimizer.step()  # 梯度下降