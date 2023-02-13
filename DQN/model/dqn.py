import torch
from DQN.model.q_net import Qnet
from DQN.model.va_net import VAnet
import numpy as np
import torch.nn.functional as F
class DQN:
    ''' DQN算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device, dqn_type='VanillaDQN'):
        self.action_dim = action_dim    #
        if dqn_type == 'DuelingDQN':  # Dueling DQN采取不一样的网络框架
            self.q_net = VAnet(state_dim, hidden_dim,
                               self.action_dim).to(device)
            self.target_q_net = VAnet(state_dim, hidden_dim,
                                      self.action_dim).to(device)
        else:
            self.q_net = Qnet(state_dim, hidden_dim,
                              self.action_dim).to(device)
            self.target_q_net = Qnet(state_dim, hidden_dim,
                                     self.action_dim).to(device)
        # self.q_net = Qnet(state_dim, hidden_dim,
        #                   self.action_dim).to(device)  # Q网络,训练网络
        # # 目标网络
        # self.target_q_net = Qnet(state_dim, hidden_dim,
        #                          self.action_dim).to(device)    # 目标网络
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate) # Adam优化器
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.dqn_type = dqn_type
        self.device = device

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()  # 这里不用调用forward，而是直接调用getitem用括号就可以完成前向传播
            # item直接把tensor变成正常的数,argmax给出的max的index
        return action

    def max_q_value(self, state):
        # state转为tensor并放入cuda
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        return self.q_net(state).max().item()   # 做一次前向传播，得到训练网络的Qsa，选取最大的动作
    def update(self, transition_dict):
        # 将参数放入torch
        # 这里的view就是tensor的reshape，第一维是batch_size，第二维保持1，为一个列
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
        # gather的作用是根据 index ，在 input 的 dim 维度上收集 value，得到所有的Qsa
        # 做batch_size次前向传播,结果是batchsize个两个action对应的输出
        # 利用gather的方式找到state分别对应的action得到的Qsa
        q_values = self.q_net(states).gather(1, actions)  # Q值
        if self.dqn_type == 'DoubleDQN':  # DQN与Double DQN的区别
            # 通过训练网络选出max_action，[1]是在取index
            max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
            # 选择action后，使用Q_target sa来更新q_target
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        else:  # DQN的情况
            # 下个状态的最大Q值
            # 使用next_states对target_q_net进行一次前向传播，选出在第一维上最大的，也就是action中最大的action对应的Qsa
            # 后面的[0]是因为max会返回value和index，这里只取value，然后变成列的形式
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(
                -1, 1)
        # DQN的更新公式，这是使用target_q_net得到的，他是Q-learning的简单推到
        # 1-dones是因为最后一个状态不应该再更新q_targets
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones
                                                                )  # TD误差目标
        # 计算loss，是更新后的目标Qsa和使用的Qsa的MSE
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        # zero_grad是为了mini_batch训练而准备的，可以把梯度累计
        # 正常训练中，batch之间梯度没什么关系，需要清空，这需要在反向传播和梯度下降之前
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        # 反向传播
        dqn_loss.backward()  # 反向传播更新参数，得到每个W的梯度值，tensor的梯度将会累加到它的.grad属性里面去
        self.optimizer.step()   # 使用梯度更新参数值
        # 累积到一定次数，target_q_net和q_net保持一次统一，也就是慢更新
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1