import numpy as np
from collections import deque
import random
import os
from game import Board, Game
from DQN import DQN, Transition
import torch
import math


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def add(self, transition):
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)


class DQNTrainer:
    def __init__(self, board_width=6, board_height=6):
        self.board_width = board_width
        self.board_height = board_height
        self.board = Board(width=board_width, height=board_height, n_in_row=4)
        self.game = Game(self.board)
        self.dqn = DQN(board_size=board_width)

        # 训练参数
        self.buffer_size = 50000
        self.batch_size = 512
        self.data_buffer = PrioritizedReplayBuffer(self.buffer_size)
        self.episodes = 10000
        self.save_freq = 100
        self.best_win_rate = 0.0

        # 奖惩参数
        self.win_reward = 10.0
        self.lose_reward = -10.0
        self.draw_reward = 1.0
        self.danger_reward = -2.0
        self.illegal_reward = -5.0
        self.step_penalty = -0.1
        self.pattern_rewards = {
            'open_three': 1.0,
            'half_four': 2.0,
            'open_four': 5.0
        }

    def get_state_representation(self, board_state):
        """获取4通道状态表示"""
        square_state = np.zeros((4, self.board_width, self.board_height), dtype=np.float32)

        if board_state:
            moves, players = np.array(list(zip(*board_state.items())))
            move_curr = moves[players == self.board.current_player]
            move_oppo = moves[players != self.board.current_player]

            square_state[0] = self._create_plane(move_curr, self.board_width, self.board_height)
            square_state[1] = self._create_plane(move_oppo, self.board_width, self.board_height)
            square_state[2] = self._create_plane([self.board.last_move], self.board_width, self.board_height)

            if len(board_state) % 2 == 0:
                square_state[3] = 1.0

        return square_state

    def _create_plane(self, moves, width, height):
        """创建单个通道的平面"""
        plane = np.zeros((width, height), dtype=np.float32)
        for move in moves:
            row, col = divmod(move, width)
            plane[row][col] = 1.0
        return plane

    def _check_patterns(self, player):
        """检查棋型模式"""
        patterns = {
            'open_three': 0,
            'half_four': 0,
            'open_four': 0
        }

        opponent = 2 if player == 1 else 1
        board_array = np.zeros((self.board_width, self.board_height))

        for move, p in self.board.states.items():
            row, col = divmod(move, self.board_width)
            board_array[row, col] = p

        # 简化的模式检查
        for row in range(self.board_width):
            for col in range(self.board_height):
                if board_array[row, col] == player:
                    for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                        count = 1
                        for i in range(1, 5):
                            x, y = row + i * dx, col + i * dy
                            if 0 <= x < self.board_width and 0 <= y < self.board_height:
                                if board_array[x, y] == player:
                                    count += 1
                                else:
                                    break

                        if count >= 3:
                            patterns['open_three'] += 1
                        if count >= 4:
                            patterns['open_four'] += 1

        return patterns

    def collect_selfplay_data(self):
        """收集自我对弈数据"""
        self.board.init_board()
        episode_data = []
        step_count = 0

        while True:
            current_state = self.get_state_representation(self.board.states)
            available_moves = self.board.availables
            player = self.board.current_player

            # 检查危险局面和棋型
            is_dangerous = False
            patterns = self._check_patterns(3 - player)

            # 选择动作
            move = self.dqn.act(current_state, available_moves)

            # 计算基础奖励
            base_reward = 0.0
            if move not in available_moves:
                reward = self.illegal_reward
                done = True
            else:
                self.board.do_move(move)
                step_count += 1
                end, winner = self.board.game_end()

                # 棋型奖励
                for pattern, cnt in patterns.items():
                    base_reward += self.pattern_rewards.get(pattern, 0) * cnt

                if end:
                    if winner == 1:
                        reward = base_reward + self.win_reward
                    elif winner == 2:
                        reward = base_reward + self.lose_reward
                    else:
                        reward = base_reward + self.draw_reward
                    done = True
                else:
                    reward = base_reward + (self.danger_reward if is_dangerous else 0) + self.step_penalty
                    done = False

            next_state = self.get_state_representation(self.board.states)
            transition = Transition(current_state, move, reward, next_state, done)
            episode_data.append(transition)

            if done:
                for i, trans in enumerate(episode_data):
                    if i < len(episode_data) - 1:
                        episode_data[i] = trans._replace(
                            reward=trans.reward + 0.9 * episode_data[i + 1].reward
                        )
                    self.data_buffer.add(episode_data[i])
                break

        return step_count

    def train_batch(self):
        """训练一个批次"""
        if len(self.data_buffer) < self.batch_size:
            return None

        beta = min(1.0, 0.4 + self.dqn.steps * (1.0 - 0.4) / 100000)
        batch, indices, weights = self.data_buffer.sample(self.batch_size, beta)

        loss, errors = self.dqn.train_step(batch, weights)
        self.data_buffer.update_priorities(indices, errors)

        # 定期保存和评估
        if self.dqn.steps % self.save_freq == 0:
            win_rate = self.evaluate_model()
            if win_rate > self.best_win_rate:
                self.best_win_rate = win_rate
                self.dqn.save_model('dqn_best.pth')
                print(f"New best model saved with win rate: {win_rate:.2f}")

        return loss

    def evaluate_model(self, n_games=10):
        """评估当前模型对随机策略的胜率"""
        win_cnt = 0
        for _ in range(n_games):
            self.board.init_board()
            while True:
                current_state = self.get_state_representation(self.board.states)
                available_moves = self.board.availables

                if self.board.current_player == 1:
                    move = self.dqn.act(current_state, available_moves, training=False)
                else:
                    move = random.choice(available_moves)

                self.board.do_move(move)
                end, winner = self.board.game_end()
                if end:
                    if winner == 1:
                        win_cnt += 1
                    break
        return win_cnt / n_games

    def run(self):
        """主训练循环"""
        for episode in range(1, self.episodes + 1):
            steps = self.collect_selfplay_data()
            loss = self.train_batch()

            if episode % 10 == 0:
                epsilon = self.dqn.epsilon_final + (self.dqn.epsilon_start - self.dqn.epsilon_final) * \
                          math.exp(-1. * self.dqn.steps / self.dqn.epsilon_decay)
                print(f"Episode {episode}, Steps: {steps}, Loss: {loss if loss else 'N/A'}, "
                      f"Epsilon: {epsilon:.3f}")

        self.dqn.save_model('dqn_final.pth')


if __name__ == "__main__":
    trainer = DQNTrainer()
    trainer.run()