import torch.nn.functional as F
import torch
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # 添加了一个softmax，这是因为输出是一个概率分布(策略)而不是Q
        return F.softmax(self.fc2(x), dim=1)