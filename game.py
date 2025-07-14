import numpy as np


class Board(object):
    def __init__(self,width,height,n_in_row):
        self.width = width
        self.height = height
        self.states = {}  #字典，记录每个位置上的玩家，记录棋盘现在的状态，键是位置，职为玩家编号
        self.n_in_row =n_in_row
        self.players = [1, 2]  # 定义玩家编号 数组

    def init_board(self, start_player=0):#初始化键盘
        self.current_player = self.players[start_player]  #表示起始玩家，  一个数组，0表示玩家1，1表示玩家2
        self.availables = list(range(self.width * self.height))   #初始化可以落子的所有空位，列表的范围是8*8=64，列表长度，所有可能点的个数，初始为0？
        self.states = {}  #清空棋盘，初始化棋盘
        self.last_move = -1 #初始上一步为无效值

    def current_state(self):  #得到当前棋盘的状态
        square_state = np.zeros((4, self.width, self.height))
        if self.states: #有棋子的话，得到相应的状态，棋盘状态不可以为空
            locations, players = np.array(list(zip(*self.states.items())))
            locas_curr = locations[players == self.current_player]
            locas_oppo = locations[players != self.current_player]   # 对手的落子位置
            square_state[0][locas_curr // self.width,
                           locas_curr % self.width] = 1.0
            square_state[1][locas_oppo // self.width,
                            locas_oppo % self.width] = 1.0

            square_state[2][self.last_move // self.width,
                            self.last_move % self.width] = 1.0   #self.last_move = 4 表示上一步落子在棋盘正中央（第1行第1列）
        if len(self.states) % 2 == 0:  #落子的个数
            square_state[3][:, :] = 1.0 #当前玩家标识 1表示黑棋，0表示白棋
        return square_state[:, ::-1, :]
    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)  # 移除可下棋的位置
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )# 将当前的player变换
        self.last_move = move #上一步操作变成现在的位置

    def get_win(self):
        width = self.width #宽
        height = self.height
        states = self.states
        n = self.n_in_row

        locaed = list(set(range(width * height)) - set(self.availables))
        if len(locaed) < self.n_in_row * 2 - 1:
            return False, -1

        for m in locaed:  # 遍历每一个落子
            h = m // width
            w = m % width
            player = states[m]

            # 检查四个方向
            directions = [
                (0, 1),  # 水平 第一个代表x轴，表示行，第二个表示列，向下加1，向右加1  行不变，列加1
                (1, 0),  # 垂直
                (1, 1),  # 主对角线
                (1, -1)  # 副对角线
            ]
            for dh, dw in directions: #dh代表第一个，dw代表第二个
                count = 1
                for i in range(1, n):
                    nh, nw = h + i * dh, w + i * dw  #首先从这一个向给出的4个方向进行检查，每次向同一个方向进行。
                    if not (0 <= nh < height and 0 <= nw < width):
                        break
                    if states.get(nh * width + nw, -1) != player:  # 如果这个位置没有落子找不到这个键，返回-1，
                        break
                    count += 1
                if count >= n:
                    return True, player
        return False, -1

    def game_end(self):
        win, winner = self.get_win()
        if win:
            return True, winner
        elif not len(self.availables):  # 没有空的位置，平局，返回-1.
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


class Game(object):

    def __init__(self, board , **kwargs):
        self.board = board  #加载棋盘

    def start_play(self, player1, player2, start_player=0):
        self.board.init_board(start_player)
        p1, p2 = self.board.players  #p1=1 ，p2=2
        player1.set_player_ind(p1) #设置玩家1代表 1
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}  #创建玩家字典
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]  #这个players指的是player1字典
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            end, winner = self.board.game_end()
            if end:
                return winner

    def start_self_play(self, player, temp=1e-3):

        self.board.init_board()
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            self.board.do_move(move)
            end, winner = self.board.game_end()  #判断游戏是否结束
            if end:
                winners_z = np.zeros(len(current_players))  #给出一个winners标志
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                player.reset_player()  #重置玩家
                return winner, zip(states, mcts_probs, winners_z)  #返回胜者，状态，每个状态的动作概率，胜者标签