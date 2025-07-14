import random
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from game import Board, Game
from MCTS import MCTSPlayer
from Policy_net import PolicyValueNet  # Pytorch


class TrainPipeline:
    def __init__(self, init_model=None):
        self.board_width = 6
        self.board_height = 6
        self.n_in_row = 4
        self.board = Board(self.board_width,
                           self.board_height,
                           self.n_in_row)
        self.game = Game(self.board)

        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0
        self.temp = 1.0
        self.temp_decay = 0.995
        self.min_temp = 0.1
        self.n_playout = 400
        self.c_puct = 5
        self.buffer_size = 10000 #缓冲区大小
        self.batch_size = 512  # 训练批次大小
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5
        self.kl_targ = 0.02
        self.check_freq = 50
        self.save_freq = 50
        self.game_batch_num = 1500
        self.best_win_ratio = 0.0


        self.loss_history = []
        self.entropy_history = []
        self.kl_history = []
        self.win_ratios = []

        if init_model:
            print("111")
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   model_file=init_model)
            self.best_policy_net = PolicyValueNet(self.board_width,
                                                  self.board_height,
                                                  model_file=init_model)
        else:
            print(222)
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height)
            self.best_policy_net = PolicyValueNet(self.board_width,
                                                  self.board_height)


        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1,
                                      noise_ratio=0.25)

    def get_equi_data(self, play_data):
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_prob.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))

                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        # 调整温度参数
        self.temp = max(self.min_temp, self.temp * self.temp_decay)
        self.mcts_player.set_temp(self.temp)

        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player,
                                                          temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)

            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = np.array([data[2] for data in mini_batch])
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)

        loss_history = []
        entropy_history = []
        kl_history = []

        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                self.learn_rate * self.lr_multiplier)

            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),axis=1))

            loss_history.append(loss)
            entropy_history.append(entropy)
            kl_history.append(kl)

            if kl > self.kl_targ * 4:
                break

        self.loss_history.append(np.mean(loss_history))
        self.entropy_history.append(np.mean(entropy_history))
        self.kl_history.append(np.mean(kl_history))

        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 - np.var(winner_batch - old_v.flatten()) /
                             np.var(winner_batch))
        explained_var_new = (1 - np.var(winner_batch - new_v.flatten()) /
                             np.var(winner_batch))

        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(np.mean(kl_history),
                        self.lr_multiplier,
                        np.mean(loss_history),
                        np.mean(entropy_history),
                        explained_var_old,
                        explained_var_new))
        return np.mean(loss_history), np.mean(entropy_history)

    def plot_training_stats(self, iteration):
        plt.figure(figsize=(12, 10))

        plt.subplot(4, 1, 1)
        plt.plot(self.loss_history, label='Loss')
        plt.title(f'Training Stats (Iteration {iteration})')
        plt.ylabel('Loss')
        plt.grid(True)

        plt.subplot(4, 1, 2)
        plt.plot(self.entropy_history, label='Entropy', color='orange')
        plt.ylabel('Entropy')
        plt.grid(True)

        plt.subplot(4, 1, 3)
        plt.plot(self.kl_history, label='KL Divergence', color='green')
        plt.ylabel('KL Divergence')
        plt.grid(True)

        plt.subplot(4, 1, 4)
        plt.plot(self.win_ratios, label='Win Ratio', color='red')
        plt.xlabel('Evaluation Interval')
        plt.ylabel('Win Ratio')
        plt.ylim(0, 1)
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f'training_stats_iter_{iteration}.png')
        plt.close()

    def policy_evaluate(self, n_games=50):
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout,
                                         noise_ratio=0.1)
        best_mcts_player = MCTSPlayer(self.best_policy_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player,
                                          best_mcts_player,
                                          start_player=i % 2)
            win_cnt[winner] += 1
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        self.win_ratios.append(win_ratio)
        print("\nEvaluation results:")
        print("num_playouts:{}, win: {}, lose: {}, tie:{}, win_ratio: {:.2f}%".format(
            self.n_playout,
            win_cnt[1], win_cnt[2], win_cnt[-1], win_ratio * 100))
        return win_ratio

    def run(self):
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print(f"batch i:{i + 1}, episode_len:{self.episode_len}, temp:{self.temp:.3f}")

                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()

                # Evaluate win ratio every 10 iterations
                if (i + 1) % self.check_freq == 0:
                    print("\nEvaluating current model...")
                    win_ratio = self.policy_evaluate()

                    if win_ratio >= 0.5:
                        print("\nNew best policy achieved!")
                        print(f"Previous best win ratio: {self.best_win_ratio * 100:.2f}%")
                        print(f"New win ratio: {win_ratio * 100:.2f}%")
                        self.best_win_ratio = win_ratio

                        try:
                            self.best_policy_net = PolicyValueNet(self.board_width,
                                                                  self.board_height,
                                                                  model_file='./current_policy.model')
                        except FileNotFoundError:
                            # 如果文件不存在，使用当前网络参数初始化
                            self.best_policy_net = PolicyValueNet(self.board_width,
                                                                  self.board_height)
                            self.best_policy_net.policy_value_net.load_state_dict(
                                self.policy_value_net.policy_value_net.state_dict()
                            )

                        self.best_policy_net.save_model('./best_policy.model')
                        print("Best model saved!\n")

                # 原有保存逻辑（每save_freq次保存当前模型）
                if (i + 1) % self.save_freq == 0:
                    print("\nSaving current model...")
                    self.policy_value_net.save_model('./current_policy.model')
                    self.plot_training_stats(i + 1)
                    print("Current model saved")

                # 新增：每100次额外保存一个历史模型（不覆盖）
                if (i + 1) % 100 == 0:
                    history_model_path = f'./history_policy_{i + 1}.model'
                    self.policy_value_net.save_model(history_model_path)
                    print(f"Additional history model saved as {history_model_path}")

            # 训练结束后保存最终模型
            final_model_path = './final_policy.model'
            self.policy_value_net.save_model(final_model_path)
            print(f"\nTraining completed! Final model saved as {final_model_path}")

        except KeyboardInterrupt:
            # 中断时也保存当前模型
            interrupt_model_path = './interrupted_policy.model'
            self.policy_value_net.save_model(interrupt_model_path)
            print(f'\nTraining interrupted! Current model saved as {interrupt_model_path}')

if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    training_pipeline.run()