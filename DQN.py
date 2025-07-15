import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import copy
import os
from collections import namedtuple, deque
import math

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class DQN(nn.Module):
    def __init__(self, board_size=6):
        super(DQN, self).__init__()
        self.board_size = board_size

        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 主网络结构
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # 双头DQN结构
        self.value_head = nn.Sequential(
            nn.Linear(256 * board_size * board_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.advantage_head = nn.Sequential(
            nn.Linear(256 * board_size * board_size, 512),
            nn.ReLU(),
            nn.Linear(512, board_size * board_size)
        )

        # 目标网络
        self.target_net = copy.deepcopy(self)
        self.target_net.eval()

        # 将模型移到设备上
        self.to(self.device)
        self.target_net.to(self.device)

        # 训练参数
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
        self.loss_fn = nn.SmoothL1Loss()

        # 探索参数
        self.epsilon_start = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 10000
        self.steps = 0

        # 优先级经验回放参数
        self.beta_start = 0.4
        self.beta_frames = 100000

        # 目标网络更新频率
        self.update_target_freq = 1000

    def forward(self, x):
        # 处理输入数据
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)

        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        if x.shape[1] != 4:
            x = x.view(-1, 4, self.board_size, self.board_size)

        # 前向传播
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)

        value = self.value_head(x)
        advantages = self.advantage_head(x)

        # 组合价值和优势
        qvals = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals

    def act(self, state, available_actions, training=True):
        #改进策略
        state = self._preprocess_state(state)

        # 动态epsilon
        epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                  math.exp(-1. * self.steps / self.epsilon_decay)

        if training and random.random() < epsilon:
            return random.choice(available_actions)

        with torch.no_grad():
            q_values = self.forward(state)

        q_values = q_values.squeeze().cpu().numpy()

        # 屏蔽非法动作
        mask = np.ones_like(q_values) * -np.inf
        mask[available_actions] = q_values[available_actions]

        if training:
            # Boltzmann探索
            temp = max(0.1, 1 - self.steps / 200000)
            exp_values = np.exp((mask - np.max(mask)) / temp)
            probs = exp_values / np.sum(exp_values)
            return np.random.choice(len(probs), p=probs)
        else:
            return np.argmax(mask)

    def train_step(self, batch, weights=None):
        #经验放回
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        if weights is not None:
            weights = torch.FloatTensor(weights).to(self.device)

        # 计算当前Q值
        current_q = self.forward(states).gather(1, actions.unsqueeze(1))

        # 使用目标网络计算目标Q值
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * 0.99 * next_q

        # 计算损失
        losses = self.loss_fn(current_q.squeeze(), target_q)

        if weights is not None:
            losses = (losses * weights).mean()
        else:
            losses = losses.mean()

        # 计算优先级
        errors = torch.abs(current_q.squeeze() - target_q).detach().cpu().numpy()

        self.optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10)
        self.optimizer.step()

        # 更新目标网络
        self.steps += 1
        if self.steps % self.update_target_freq == 0:
            # 软更新目标网络
            for target_param, param in zip(self.target_net.parameters(), self.parameters()):
                target_param.data.copy_(0.01 * param.data + 0.99 * target_param.data)

        return losses.item(), errors

    def _preprocess_state(self, state):
        #状态处理
        if isinstance(state, np.ndarray):
            if state.ndim == 2:
                state = np.stack([state] * 4)
            state = np.ascontiguousarray(state[np.newaxis, ...])
            return torch.FloatTensor(state).to(self.device)
        return state

    def save_model(self, filename):
        torch.save({
            'state_dict': self.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'steps': self.steps
        }, filename)

    def load_model(self, filename):
        if os.path.exists(filename):
            checkpoint = torch.load(filename)
            self.load_state_dict(checkpoint['state_dict'])
            self.target_net.load_state_dict(checkpoint['target_state_dict'])
            self.steps = checkpoint.get('steps', 0)
            self.eval()
