%%capture
# ensure we are on the latest version of kaggle-environments
!pip install --upgrade kaggle-environments


# Now let's set up the chess environment!
from kaggle_environments import make
env = make("chess", debug=True)


%%writefile submission.py
import time, math, random
from itertools import count
from collections import namedtuple

version = "sunfish 2025"

###############################################################################
# Piece-Square Tables 与棋子基本价值（用于评估函数）
###############################################################################

piece = {"P": 100, "N": 280, "B": 320, "R": 479, "Q": 929, "K": 60000}
pst = {
    'P': (   0,   0,   0,   0,   0,   0,   0,   0,
            78,  83,  86,  73, 102,  82,  85,  90,
             7,  29,  21,  44,  40,  31,  44,   7,
           -17,  16,  -2,  15,  14,   0,  15, -13,
           -26,   3,  10,   9,   6,   1,   0, -23,
           -22,   9,   5, -11, -10,  -2,   3, -19,
           -31,   8,  -7, -37, -36, -14,   3, -31,
             0,   0,   0,   0,   0,   0,   0,   0),
    'N': ( -66, -53, -75, -75, -10, -55, -58, -70,
            -3,  -6, 100, -36,   4,  62,  -4, -14,
            10,  67,   1,  74,  73,  27,  62,  -2,
            24,  24,  45,  37,  33,  41,  25,  17,
            -1,   5,  31,  21,  22,  35,   2,   0,
           -18,  10,  13,  22,  18,  15,  11, -14,
           -23, -15,   2,   0,   2,   0, -23, -20,
           -74, -23, -26, -24, -19, -35, -22, -69),
    'B': ( -59, -78, -82, -76, -23,-107, -37, -50,
           -11,  20,  35, -42, -39,  31,   2, -22,
            -9,  39, -32,  41,  52, -10,  28, -14,
            25,  17,  20,  34,  26,  25,  15,  10,
            13,  10,  17,  23,  17,  16,   0,   7,
            14,  25,  24,  15,   8,  25,  20,  15,
            19,  20,  11,   6,   7,   6,  20,  16,
            -7,   2, -15, -12, -14, -15, -10, -10),
    'R': (  35,  29,  33,   4,  37,  33,  56,  50,
            55,  29,  56,  67,  55,  62,  34,  60,
            19,  35,  28,  33,  45,  27,  25,  15,
             0,   5,  16,  13,  18,  -4,  -9,  -6,
           -28, -35, -16, -21, -13, -29, -46, -30,
           -42, -28, -42, -25, -25, -35, -26, -46,
           -53, -38, -31, -26, -29, -43, -44, -53,
           -30, -24, -18,   5,  -2, -18, -31, -32),
    'Q': (   6,   1,  -8,-104,  69,  24,  88,  26,
            14,  32,  60, -10,  20,  76,  57,  24,
            -2,  43,  32,  60,  72,  63,  43,   2,
             1, -16,  22,  17,  25,  20, -13,  -6,
           -14, -15,  -2,  -5,  -1, -10, -20, -22,
           -30,  -6, -13, -11, -16, -11, -16, -27,
           -36, -18,   0, -19, -15, -15, -21, -38,
           -39, -30, -31, -13, -31, -36, -34, -42),
    'K': (   4,  54,  47, -99, -99,  60,  83, -62,
           -32,  10,  55,  56,  56,  55,  10,   3,
           -62,  12, -57,  44, -67,  28,  37, -31,
           -55,  50,  11,  -4, -19,  13,   0, -49,
           -55, -43, -52, -28, -51, -47,  -8, -50,
           -47, -42, -43, -79, -64, -32, -29, -32,
            -4,   3, -14, -50, -57, -18,  13,   4,
            17,  30,  -3, -14,   6,  -1,  40,  18),
}
# 将每个表 pad（前后各 20 个 0 ）使得棋盘边界判断更快
for k, table in pst.items():
    padrow = lambda row: (0,) + tuple(x + piece[k] for x in row) + (0,)
    pst[k] = sum((padrow(table[i * 8 : i * 8 + 8]) for i in range(8)), ())
    pst[k] = (0,) * 20 + pst[k] + (0,) * 20

###############################################################################
# 全局常量与初始局面
###############################################################################

# 采用 120 格字符串表示棋盘，边界用空格填充，便于检测非法走子
A1, H1, A8, H8 = 91, 98, 21, 28
initial = (
    "         \n"  #  0–9
    "         \n"  # 10–19
    " rnbqkbnr\n"  # 20–29
    " pppppppp\n"  # 30–39
    " ........\n"  # 40–49
    " ........\n"  # 50–59
    " ........\n"  # 60–69
    " ........\n"  # 70–79
    " PPPPPPPP\n"  # 80–89
    " RNBQKBNR\n"  # 90–99
    "         \n"  #100–109
    "         \n"  #110–119
)

# 各方向偏移量（10×8棋盘编码中，上、右、下、左分别为 -10, +1, +10, -1）
N, E, S, W = -10, 1, 10, -1
directions = {
    "P": (N, N+N, N+W, N+E),
    "N": (N+N+E, E+N+E, E+S+E, S+S+E, S+S+W, W+S+W, W+N+W, N+N+W),
    "B": (N+E, S+E, S+W, N+W),
    "R": (N, E, S, W),
    "Q": (N, E, S, W, N+E, S+E, S+W, N+W),
    "K": (N, E, S, W, N+E, S+E, S+W, N+W)
}

# Mate 值设置，确保评估值能区分 mate 与普通局面
MATE_LOWER = piece["K"] - 10 * piece["Q"]
MATE_UPPER = piece["K"] + 10 * piece["Q"]

###############################################################################
# 棋局状态与走法生成
###############################################################################

Move = namedtuple("Move", "i j prom")

class Position(namedtuple("Position", "board score wc bc ep kp")):
    """
    board -- 120字符表示的棋盘（含边界）
    score -- 当前局面评估值
    wc, bc -- 各自的王车易位权，例如 (True, True) 表示左右均可
    ep -- 可走 en passant 的格子编号（否则为 0）
    kp -- 棋王辅助（如在 castling 时用到），否则为 0
    """
    def gen_moves(self):
        for i, p in enumerate(self.board):
            if not p.isupper():
                continue
            for d in directions[p]:
                for j in count(i + d, d):
                    q = self.board[j]
                    # 超出棋盘或遇到己方棋子则中断该“射线”
                    if q.isspace() or q.isupper():
                        break
                    # 对于兵：单走、双走、吃子及升变
                    if p == "P":
                        if d in (N, N+N) and q != ".": 
                            break
                        if d == N+N and (i < A1+N or self.board[i+N] != "."):
                            break
                        if d in (N+W, N+E) and q == "." and j not in (self.ep, self.kp, self.kp-1, self.kp+1):
                            break
                        if A8 <= j <= H8:
                            for prom in "NBRQ":
                                yield Move(i, j, prom)
                            break
                    yield Move(i, j, "")
                    # 对于兵、马、王：走一步后停止；吃子后也停止
                    if p in "PNK" or q.islower():
                        break
                    # 对于王车易位：检查并生成对应走法
                    if i == A1 and self.board[j+E] == "K" and self.wc[0]:
                        yield Move(j+E, j+W, "")
                    if i == H1 and self.board[j+W] == "K" and self.wc[1]:
                        yield Move(j+W, j+E, "")
                        
    def rotate(self, nullmove=False):
        """将棋盘旋转（换边），同时将评估值取负"""
        return Position(
            self.board[::-1].swapcase(), -self.score, self.bc, self.wc,
            119 - self.ep if self.ep and not nullmove else 0,
            119 - self.kp if self.kp and not nullmove else 0,
        )
    
    def move(self, move):
        i, j, prom = move
        p, q = self.board[i], self.board[j]
        put = lambda board, i, p: board[:i] + p + board[i+1:]
        board = self.board
        wc, bc, ep, kp = self.wc, self.bc, 0, 0
        score = self.score + self.value(move)
        # 执行走子：把源格的棋子移动到目标格
        board = put(board, j, board[i])
        board = put(board, i, ".")
        # 更新易位权
        if i == A1: wc = (False, wc[1])
        if i == H1: wc = (wc[0], False)
        if j == A8: bc = (bc[0], False)
        if j == H8: bc = (False, bc[1])
        # 王移动时禁用易位；若是易位，则移动对应的车
        if p == "K":
            wc = (False, False)
            if abs(j - i) == 2:
                kp = (i + j) // 2
                board = put(board, A1 if j < i else H1, ".")
                board = put(board, kp, "R")
        # 兵：升变、双步走、en passant 捕获
        if p == "P":
            if A8 <= j <= H8:
                board = put(board, j, prom)
            if j - i == 2 * N:
                ep = i + N
            if j == self.ep:
                board = put(board, j+S, ".")
        return Position(board, score, wc, bc, ep, kp).rotate()
    
    def value(self, move):
        i, j, prom = move
        p, q = self.board[i], self.board[j]
        score = pst[p][j] - pst[p][i]
        if q.islower():
            score += pst[q.upper()][119 - j]
        if abs(j - self.kp) < 2:
            score += pst["K"][119 - j]
        if p == "K" and abs(i - j) == 2:
            score += pst["R"][(i + j) // 2]
            score -= pst["R"][A1 if j < i else H1]
        if p == "P":
            if A8 <= j <= H8:
                score += pst[prom][j] - pst["P"][j]
            if j == self.ep:
                score += pst["P"][119 - (j+S)]
        return score

###############################################################################
# 采用 α–β 剪枝的 Negamax 搜索（递归版）
###############################################################################

def alphabeta(pos, depth, alpha, beta):
    # 深度为 0 时返回当前局面评估值（无需返回走法）
    if depth == 0:
        return pos.score, None
    best_move = None
    moves = list(pos.gen_moves())
    # 如果没有走法，则直接返回局面评估值
    if not moves:
        return pos.score, None
    # 按走法“价值”降序排列，有助于剪枝
    moves.sort(key=lambda m: pos.value(m), reverse=True)
    for move in moves:
        # 递归调用：注意这里使用 negamax 思路，返回值取负
        score, _ = alphabeta(pos.move(move), depth - 1, -beta, -alpha)
        score = -score
        if score > alpha:
            alpha = score
            best_move = move
        if alpha >= beta:
            break
    return alpha, best_move

###############################################################################
# UCI 辅助函数：将棋盘位置和走法格式相互转换
###############################################################################

def parse(c):
    fil = ord(c[0]) - ord("a")
    rank = int(c[1]) - 1
    return A1 + fil - 10 * rank

def render(i):
    rank, fil = divmod(i - A1, 10)
    return chr(fil + ord("a")) + str(-rank + 1)

def fen_to_position(fen):
    # fen 格式：棋子布局、执棋方、易位权、兵可过路格、半回合钟、全回合数
    places, my_color, castling, en_passant, hm_clock, fm_number = fen.split()
    blank_lines = "         \n" * 2
    pos = blank_lines + " "
    for c in places:
        if c == "/":
            pos += "\n "
            continue
        try:
            pos += "." * int(c)
        except ValueError:
            pos += c
    pos += "\n" + blank_lines
    white_castling = (("Q" in castling), ("K" in castling))
    black_castling = (("k" in castling), ("q" in castling))
    en_passant = 0 if en_passant == "-" else parse(en_passant)
    position = Position(pos, 0, white_castling, black_castling, en_passant, 0)
    if my_color == "b":
        position = position.rotate()
    print("Loaded position")
    return my_color, position

###############################################################################
# Chess Bot 主函数
###############################################################################

from Chessnut import Game

def chess_bot(obs):
    # 先用 Chessnut 获取所有合法走法（以防万一搜索失败时）
    game = Game(obs.board)
    moves = list(game.get_moves())
    think_time = 0.1  # 思考时间（秒）
    start = time.time()
    my_color, position = fen_to_position(obs.board)
    best_move = None
    max_depth = 1
    # 采用迭代加深，直到用时超限
    while time.time() - start < think_time:
        score, move = alphabeta(position, max_depth, -MATE_UPPER, MATE_UPPER)
        if move is not None:
            best_move = move
        max_depth += 1
    if best_move is None:
        return random.choice(moves)
    i, j = best_move.i, best_move.j
    if my_color == "b":
        i, j = 119 - i, 119 - j
    move_str = render(i) + render(j) + best_move.prom.lower()
    if move_str in moves:
        return move_str
    else:
        print("illegal move:", move_str)
        return random.choice(moves)



result = env.run(["submission.py", "random"])
print("Agent exit status/reward/time left: ")
# look at the generated replay.json and print out the agent info
for agent in result[-1]:
    print("\t", agent.status, "/", agent.reward, "/", agent.observation.remainingOverageTime)
print("\n")
# render the game
env.render(mode="ipython", width=1000, height=1000) 

