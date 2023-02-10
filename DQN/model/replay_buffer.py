import random
import numpy as np
import collections

class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        # collection实现的一个队列
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)    # 从buffer中取出batch_size个记忆
        state, action, reward, next_state, done = zip(*transitions) # 解压元组列表，变成多个元组
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)