import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def learning_rate(optimizer, lr):#用于设置学习速率
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Net(nn.Module):
    def __init__(self, board_width, board_height):
        super(Net, self).__init__()

        self.board_width = board_width
        self.board_height = board_height

        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)   #4个1*1的卷积核降维
        self.act_fc1 = nn.Linear(4*board_width*board_height,
                                 board_width*board_height)#输出每个个位置的落子概率


        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2*board_width*board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):#激活函数
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4*self.board_width*self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act))
        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2*self.board_width*self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))
        return x_act, x_val


class PolicyValueNet():#在蒙特卡洛树搜索过程中评估叶子节点对应的局面评分和返回该局面下的所有可行动作及对应概率

    def __init__(self, board_width, board_height,
                 model_file=None, use_gpu=False):
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4
        if self.use_gpu:
            self.policy_value_net = Net(board_width, board_height).cuda()
        else:
            self.policy_value_net = Net(board_width, board_height)#不适用GPU
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),##返回模型的参数
                                    weight_decay=self.l2_const)

        if model_file:
            net_params = torch.load(model_file)#加载模型参数
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch):#一批状态

        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))#将 state_batch 转换为 PyTorch 的张量，但不移动到 GPU 上
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())
            return act_probs, value.data.numpy()
    def policy_value_fn(self, board):#用于得到状态价值以及相应的动作概率

        legal_positions = board.availables#合法位置
        current_state = np.ascontiguousarray(board.current_state().reshape(
                -1, 4, self.board_width, self.board_height))
        if self.use_gpu:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).cuda().float())
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
            value = value.data.cpu().numpy()[0][0]
        else:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).float())
            act_probs = np.exp(log_act_probs.data.numpy().flatten())
            value = value.data.numpy()[0][0]#将状态价值转换为 NumPy 数组。
        act_probs = zip(legal_positions, act_probs[legal_positions])#将合法位置和对应的动作概率组合在一起
        return act_probs, value#返回位置概率以及价值

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
            winner_batch = Variable(torch.FloatTensor(winner_batch).cuda())
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))#将状态批次转换为 PyTorch 张量。
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))#将 MCTS 概率转换为 PyTorch 张量。
            winner_batch = Variable(torch.FloatTensor(winner_batch))#将赢家批次转换为 PyTorch 张量


        self.optimizer.zero_grad()

        learning_rate(self.optimizer, lr)#设置学习率


        log_act_probs, value = self.policy_value_net(state_batch)#将状态批次输入到策略价值网络中，得到动作对数概率和状态价值。
        value_loss = F.mse_loss(value.view(-1), winner_batch)#计算价值损失，使用均方误差损失函数。
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))#计算策略损失，使用交叉熵损失函数
        loss = value_loss + policy_loss#总损失为价值损失和策略损失的和。

        loss.backward()#反向传播计算梯度。
        self.optimizer.step()#使用优化器更新参数。
        entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
                )#计算策略熵，用于监控。
        return loss.item(), entropy.item()#返回损失和熵的值。


    def save_model(self, model_file):#保存模型
        net_params = self.policy_value_net.state_dict()#得到获取策略价值网络的参数字典
        torch.save(net_params, model_file)