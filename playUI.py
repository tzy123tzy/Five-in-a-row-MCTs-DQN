import os
import pygame
import sys
import torch
from game import Board, Game
from MCTS import MCTSPlayer
from Policy_net import PolicyValueNet
from DQN import DQN
import numpy as np
# 初始化Pygame
pygame.init()

#颜色定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BOARD_COLOR = (220, 179, 92) 
LINE_COLOR = (0, 0, 0)  # 线条颜色
HIGHLIGHT_COLOR = (255, 0, 0)  
TEXT_COLOR = (19, 25, 99)  # 文本颜色
BUTTON_COLOR = (100, 100, 100)  
BUTTON_HOVER_COLOR = (150, 150, 150)  
MCTS_BUTTON_COLOR = (100, 200, 100)  #MCT按钮颜色
DQN_BUTTON_COLOR = (100, 100, 200)  # DQN按钮颜色

# 棋盘设置
BOARD_SIZE = 6
GRID_SIZE = 50  # Pixel size of each grid cell
PIECE_RADIUS = 20  # Radius of game pieces
MARGIN = 30  # Margin around the board
WINDOW_WIDTH = (BOARD_SIZE - 1) * GRID_SIZE + 2 * MARGIN
WINDOW_HEIGHT = (BOARD_SIZE - 1) * GRID_SIZE + 2 * MARGIN + 100
INFO_HEIGHT = 70  # Height of info area


class GomokuUI:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Gomoku - Human vs AI")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 20)
        self.title_font = pygame.font.SysFont('Arial', 30)
        self.small_font = pygame.font.SysFont('Arial', 16)

        # 游戏状态初始化
        self.board = None
        self.game = None
        self.ai_player = None
        self.human_player = None
        self.game_over = False
        self.winner = None
        self.in_menu = True
        self.ai_type = "MCTS"  #选择

        # Load AI models
        self.load_models()

    def load_models(self):
        # MCTS 网络
        if os.path.exists('best_policy.model'):
           self.best_policy = PolicyValueNet(BOARD_SIZE, BOARD_SIZE, model_file='best_policy.model')
        else:
           raise FileNotFoundError("MCTS model file not found")

        # DQN模型
        self.dqn_model = DQN(board_size=BOARD_SIZE)
        if os.path.exists('dqn_best.pth'):
            self.dqn_model.load_model('dqn_best.pth')

    def start_menu(self):

        title = self.title_font.render("Gomoku - Human vs AI", True, BLACK)
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 4))

        #按钮设置
        human_btn = pygame.Rect(WINDOW_WIDTH // 4, WINDOW_HEIGHT // 2, WINDOW_WIDTH // 2, 40)
        ai_btn = pygame.Rect(WINDOW_WIDTH // 4, WINDOW_HEIGHT // 2 + 50, WINDOW_WIDTH // 2, 40)
        mcts_btn = pygame.Rect(WINDOW_WIDTH // 4, WINDOW_HEIGHT // 2 + 110, WINDOW_WIDTH // 4, 30)
        dqn_btn = pygame.Rect(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 110, WINDOW_WIDTH // 4, 30)

        while self.in_menu:
            mouse_pos = pygame.mouse.get_pos()

            # 绘制菜单
            self.screen.fill(BOARD_COLOR)
            self.screen.blit(title, title_rect)

            
            ai_select_text = self.font.render("Select AI Type:", True, BLACK)
            self.screen.blit(ai_select_text, (WINDOW_WIDTH // 4, WINDOW_HEIGHT // 2 + 90))

           
            pygame.draw.rect(self.screen,
                             BUTTON_HOVER_COLOR if human_btn.collidepoint(mouse_pos) else BUTTON_COLOR,
                             human_btn)
            pygame.draw.rect(self.screen,
                             BUTTON_HOVER_COLOR if ai_btn.collidepoint(mouse_pos) else BUTTON_COLOR,
                             ai_btn)
            pygame.draw.rect(self.screen,
                             MCTS_BUTTON_COLOR if self.ai_type == "MCTS" else BUTTON_COLOR,
                             mcts_btn)
            pygame.draw.rect(self.screen,
                             DQN_BUTTON_COLOR if self.ai_type == "DQN" else BUTTON_COLOR,
                             dqn_btn)

           
            human_text = self.font.render("Human First (Black)", True, WHITE)
            ai_text = self.font.render("AI First (White)", True, WHITE)
            mcts_text = self.font.render("MCTS", True, WHITE)
            dqn_text = self.font.render("DQN", True, WHITE)

            self.screen.blit(human_text, (human_btn.centerx - human_text.get_width() // 2,
                                          human_btn.centery - human_text.get_height() // 2))
            self.screen.blit(ai_text, (ai_btn.centerx - ai_text.get_width() // 2,
                                       ai_btn.centery - ai_text.get_height() // 2))
            self.screen.blit(mcts_text, (mcts_btn.centerx - mcts_text.get_width() // 2,
                                         mcts_btn.centery - mcts_text.get_height() // 2))
            self.screen.blit(dqn_text, (dqn_btn.centerx - dqn_text.get_width() // 2,
                                        dqn_btn.centery - dqn_text.get_height() // 2))

            # 交互事件处理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left mouse button
                        if human_btn.collidepoint(mouse_pos):
                            self.start_game(human_first=True)
                        elif ai_btn.collidepoint(mouse_pos):
                            self.start_game(human_first=False)
                        elif mcts_btn.collidepoint(mouse_pos):
                            self.ai_type = "MCTS"
                        elif dqn_btn.collidepoint(mouse_pos):
                            self.ai_type = "DQN"

            pygame.display.flip()
            self.clock.tick(30)

    def start_game(self, human_first=True):
       #初始化游戏
        self.in_menu = False
        self.board = Board(width=BOARD_SIZE, height=BOARD_SIZE, n_in_row=4)
        self.game = Game(self.board)
        self.game_over = False
        self.winner = None

        # Initialize the appropriate AI player
        if self.ai_type == "MCTS":
            self.ai_player = MCTSPlayer(self.best_policy.policy_value_fn,
                                        c_puct=5,
                                        n_playout=200)
        else:  # DQN
            self.ai_player = self.create_dqn_player()

        # 判断先行者
        if human_first:
            self.human_player = 1
            if self.ai_type == "MCTS":
                self.ai_player.set_player_ind(2)
            self.board.init_board(start_player=0)  # Human first
        else:
            self.human_player = 2
            if self.ai_type == "MCTS":
                self.ai_player.set_player_ind(1)
            self.board.init_board(start_player=0)  # AI first
            self.ai_move()

    def create_dqn_player(self):
        #创建DQN模型

        def policy_fn(board):
            # 应该使用4通道状态表示，与MCTS一致
            state = np.zeros((4, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

            for move, player in board.states.items():
                row = move // BOARD_SIZE
                col = move % BOARD_SIZE
                if player == board.current_player:
                    state[0][row][col] = 1.0  # 当前玩家
                else:
                    state[1][row][col] = 1.0  # 对手

            if board.last_move != -1:
                row = board.last_move // BOARD_SIZE
                col = board.last_move % BOARD_SIZE
                state[2][row][col] = 1.0  # 最后落子

            state[3][:, :] = 1.0 if board.current_player == 1 else 0.0  # 玩家标识

            legal_moves = board.availables
            move = self.dqn_model.act(state, legal_moves)
            return move

        return policy_fn

    def draw_board(self):
        
        self.screen.fill(BOARD_COLOR)

        
        for i in range(BOARD_SIZE):
            # 水平线
            pygame.draw.line(self.screen, LINE_COLOR,
                             (MARGIN, MARGIN + i * GRID_SIZE),
                             (WINDOW_WIDTH - MARGIN, MARGIN + i * GRID_SIZE), 2)
            # 垂直线
            pygame.draw.line(self.screen, LINE_COLOR,
                             (MARGIN + i * GRID_SIZE, MARGIN),
                             (MARGIN + i * GRID_SIZE, WINDOW_HEIGHT - INFO_HEIGHT - MARGIN), 2)

        # 绘制
        for move, player in self.board.states.items():
            row = move // BOARD_SIZE
            col = move % BOARD_SIZE
            x = MARGIN + col * GRID_SIZE
            y = MARGIN + row * GRID_SIZE

            if player == 1:  # Black piece
                pygame.draw.circle(self.screen, BLACK, (x, y), PIECE_RADIUS)
            else: #白子
                pygame.draw.circle(self.screen, WHITE, (x, y), PIECE_RADIUS)
                pygame.draw.circle(self.screen, BLACK, (x, y), PIECE_RADIUS, 1)  # Border

       
        pygame.draw.rect(self.screen, WHITE, (0, WINDOW_HEIGHT - INFO_HEIGHT, WINDOW_WIDTH, INFO_HEIGHT))

        
        if self.game_over:
            if self.winner == self.human_player:
                text = "You win! Click to return to menu."
            elif self.winner == (1 if self.ai_type == "DQN" else self.ai_player.player):
                text = "AI wins! Click to return to menu."
            else:
                text = "It's a tie! Click to return to menu."
        else:
            if self.board.current_player == self.human_player:
                color = "Black" if self.human_player == 1 else "White"
                text = f"Your turn ({color})"
            else:
                color = "Black" if (self.ai_type == "DQN" and self.human_player == 2) or (
                            self.ai_type == "MCTS" and self.ai_player.player == 1) else "White"
                text = f"AI thinking ({color})..."

        text_surface = self.font.render(text, True, TEXT_COLOR)
        self.screen.blit(text_surface, (20, WINDOW_HEIGHT - INFO_HEIGHT + 20))

       
        ai_text = self.small_font.render(f"AI: {self.ai_type}", True, TEXT_COLOR)
        self.screen.blit(ai_text, (20, WINDOW_HEIGHT - INFO_HEIGHT + 50))

    def handle_click(self, pos):
        
        if self.game_over:
            self.in_menu = True
            self.start_menu()
            return

        if self.board.current_player != self.human_player:
            return  

       
        col = min(max(0, round((pos[0] - MARGIN) / GRID_SIZE)), BOARD_SIZE - 1)
        row = min(max(0, round((pos[1] - MARGIN) / GRID_SIZE)), BOARD_SIZE - 1)
        move = row * BOARD_SIZE + col

        if move in self.board.availables:
            self.board.do_move(move)
            self.draw_board()
            pygame.display.flip()

            
            end, winner = self.board.game_end()
            if end:
                self.game_over = True
                self.winner = winner
            else:
                self.ai_move()

    def ai_move(self):
        #AI移动
        if not self.game_over and (
                (self.ai_type == "MCTS" and self.board.current_player == self.ai_player.player) or
                (self.ai_type == "DQN" and self.board.current_player != self.human_player)
        ):
           
            self.draw_board()
            pygame.display.flip()

            
            if self.ai_type == "MCTS":
                move = self.ai_player.get_action(self.board)
            else:  # DQN
                move = self.ai_player(self.board)

            self.board.do_move(move)

            
            end, winner = self.board.game_end()
            if end:
                self.game_over = True
                self.winner = winner

    def run(self):
        
        self.start_menu()

        while True:
            if not self.in_menu:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:  
                            self.handle_click(event.pos)

                self.draw_board()
                pygame.display.flip()
                self.clock.tick(30)

                
                if not self.game_over and not self.in_menu:
                    if self.ai_type == "MCTS" and self.board.current_player == self.ai_player.player:
                        self.ai_move()
                    elif self.ai_type == "DQN" and self.board.current_player != self.human_player:
                        self.ai_move()
            else:
                pass  # 菜单处理


if __name__ == "__main__":
    game = GomokuUI()
    game.run()
