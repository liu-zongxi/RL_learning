from ActorCritic.model.policy_net import PolicyNet
from ActorCritic.model.value_net import ValueNet
import torch
import torch.nn.functional as F

class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 gamma, device):
        # 策略网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)  # 价值网络
        # 策略网络优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)  # 价值网络优化器
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        '''根据决策的概率分布采样出一个动作'''
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        # 时序差分目标
        # 计算TD
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        # 真正的时序差分，这被称为时序差分残差
        # 因此critic还是在对Q进行拟合
        td_delta = td_target - self.critic(states)  # 时序差分误差
        log_probs = torch.log(self.actor(states).gather(1, actions))    # 选择action对应的概率
        # actor的loss，这符合reinforce
        # 负号进行梯度累加
        # detach表明不要让td_delta参与梯度计算
        # 类似于DQN的target网络不会产生梯度来改变参数
        # 而是仅作为目标出现
        actor_loss = torch.mean(-log_probs * td_delta.detach())
        # 均方误差损失函数
        # 这和DQN是一致的，使得Q和TD没有差距
        # detach表明不要让td_target参与梯度计算，也就类似于DQN的target网络不会变
        # 我们不希望通过target来更新梯度，而只是通过Q产生的梯度来训练网络
        critic_loss = torch.mean(
            F.mse_loss(self.critic(states), td_target.detach()))
        # 清空梯度
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()  # 计算策略网络的梯度
        critic_loss.backward()  # 计算价值网络的梯度
        self.actor_optimizer.step()  # 更新策略网络的参数
        self.critic_optimizer.step()  # 更新价值网络的参数