import numpy as np
import copy


def softmax(x):  # 计算输出向量的概率分布
    probs = np.exp(x - np.max(x))  # 减去最大值便于数值的稳定
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    def __init__(self, parent, prior_p):
        self.parent = parent  # 当前节点的父节点，以及当前节点被选择的先验概率
        self.children = {}  # 记录当前节点的子节点
        self.n_visits = 0  # 记录当前节点的访问次数
        self.Q = 0  # 平均行动价值
        self.u = 0  # 置信上限
        self.P = prior_p  # 先验概率

    def expand(self, action_priors):  # 从父亲节点扩展子节点
        for action, prob in action_priors:  # 遍历列表
            if action not in self.children:
                self.children[action] = TreeNode(self, prob)

    def select(self, c_puct):  # 蒙特卡洛中的选择
        return max(self.children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):  # 更新节点
        self.n_visits += 1  # 访问次数加1
        self.Q += 1.0 * (leaf_value - self.Q) / self.n_visits  # 使用增量的方法更新Q

    def update_recursive(self, leaf_value):  # 递归更新
        if self.parent:
            self.parent.update_recursive(-leaf_value)  # 递归更新父节点
        self.update(leaf_value)

    def get_value(self, c_puct):
        self.u = (c_puct * self.P *
                   np.sqrt(self.parent.n_visits) / (1 + self.n_visits))  # 得到U的值
        return self.Q + self.u

    def is_leaf(self):
        return self.children == {}

    def is_root(self):
        return self.parent is None


class MCTS(object):
    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        self.root = TreeNode(None, 1.0)
        self.policy = policy_value_fn
        self.c_puct = c_puct  # 控制蒙特卡洛的探索程度
        self.n_playout = n_playout  # 循环执行次数

    def _playout(self, state):  # 模拟一次对局
        node = self.root
        while True:
            if node.is_leaf():
                break

            action, node = node.select(self.c_puct)  # 选择动作
            state.do_move(action)  # 执行动作

        action_probs, leaf_value = self.policy(state)  # 获取策略和价值

        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)  # 扩展节点
        else:
            if winner == -1:  # 平局
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.get_current_player() else -1.0
                )

        node.update_recursive(-leaf_value)  # 回传更新

    def get_move_probs(self, state, temp=1e-3):  # 获取动作概率
        for n in range(self.n_playout):  # 执行多次模拟
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        act_visits = [(act, node.n_visits)
                      for act, node in self.root.children.items()]  # 获取访问次数
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        if last_move in self.root.children:  # 复用子树
            self.root = self.root.children[last_move]
            self.root._parent = None
        else:
            self.root = TreeNode(None, 1.0)


class MCTSPlayer(object):
    def __init__(self, policy_value_function,
                 c_puct=5, n_playout=2000, is_selfplay=0, noise_ratio=0.25):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay
        self.noise_ratio = noise_ratio  # 噪声比例
        self.temp = 1.0  # 初始温度

    def set_player_ind(self, p):
        self.player = p

    def set_temp(self, temp):
        self.temp = temp

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=None, return_prob=0):
        if temp is None:
            temp = self.temp

        sensible_moves = board.availables
        move_probs = np.zeros(board.width * board.height)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp)
            move_probs[list(acts)] = probs

            if self._is_selfplay:
                noise = np.random.dirichlet(0.3 * np.ones(len(probs)))
                move = np.random.choice(
                    acts,
                    p=(1 - self.noise_ratio) * probs + self.noise_ratio * noise
                )
                self.mcts.update_with_move(move)
            else:
                if len(acts) > 1:
                    noise = np.random.dirichlet(0.1 * np.ones(len(probs)))
                    probs = 0.9 * probs + 0.1 * noise
                move = np.random.choice(acts, p=probs)
                self.mcts.update_with_move(-1)

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("the board is full")