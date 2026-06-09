%%capture
!pip install --upgrade kaggle-environments
!pip install kaggle-environments Chessnut torch chess python-chess


# %%writefile main.py
import chess
import numpy as np
from ctypes import cdll, c_char_p
import os
import time
from kaggle_environments import make

class StockfishNNUE:
    """Stockfish NNUE评估器"""
    def __init__(self, nnue_path="/kaggle/input/nnue/other/default/1/nn-c3ca321c51c9.nnue"):
        # 加载NNUE动态库
        try:
            self.nnue = cdll.LoadLibrary("/kaggle/input/nnue/other/default/1/libnnueprobe.so")
            self.nnue.nnue_init(nnue_path.encode('utf-8'))
        except:
            print("警告: 无法加载NNUE库，将使用基础评估函数")
            self.nnue = None
            
        # 增加置换表大小限制
        self.tt_size = 100000  # 减小置换表大小以加快访问速度
        self.tt = {}
        
        # 增加时间管理参数
        self.nodes = 0
        self.start_time = 0
        self.time_limit = 0
        
        # 历史启发表
        self.history_table = {}
        
        # 杀手着法表
        self.killer_moves = [[None, None] for _ in range(100)]
        
        # 开局库
        self.opening_book = self._init_opening_book()
        
    def _init_opening_book(self):
        """初始化一个简单的开局库"""
        book = {
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1": [
                "e2e4", "d2d4", "c2c4"  # 主要的开局着法
            ]
        }
        return book
        
    def evaluate_position(self, board):
        """评估局面分数"""
        # 终局评估
        if board.is_checkmate():
            return -10000 if board.turn else 10000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
            
        # 使用NNUE评估
        if self.nnue is not None:
            try:
                return self.nnue.nnue_evaluate_fen(board.fen().encode('utf-8'))
            except:
                pass
                
        # 后备的基础评估
        return self._basic_evaluate(board)
        
    def _basic_evaluate(self, board):
        """基础局面评估"""
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        # 子力价值
        score = 0
        for piece_type in piece_values:
            score += len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
            score -= len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]
            
        # 位置价值(简化版)
        pawn_table = [
            0,  0,  0,  0,  0,  0,  0,  0,
            50, 50, 50, 50, 50, 50, 50, 50,
            10, 10, 20, 30, 30, 20, 10, 10,
            5,  5, 10, 25, 25, 10,  5,  5,
            0,  0,  0, 20, 20,  0,  0,  0,
            5, -5,-10,  0,  0,-10, -5,  5,
            5, 10, 10,-20,-20, 10, 10,  5,
            0,  0,  0,  0,  0,  0,  0,  0
        ]
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                if piece.piece_type == chess.PAWN:
                    if piece.color:
                        score += pawn_table[square]
                    else:
                        score -= pawn_table[63 - square]
                        
        return score
        
    def is_time_up(self):
        """检查是否超时"""
        return self.time_limit and (time.time() - self.start_time) > self.time_limit
        
    def quiescence_search(self, board, alpha, beta, depth=0):
        """简化的静态搜索"""
        if self.is_time_up():
            return alpha
            
        stand_pat = self.evaluate_position(board)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
            
        if depth <= -2:  # 减少静态搜索深度
            return stand_pat
            
        # 只考虑有价值的吃子着法
        for move in board.legal_moves:
            if not board.is_capture(move):
                continue
                
            # 快速评估吃子价值
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            if victim and attacker and victim.piece_type <= attacker.piece_type:
                continue
                
            board.push(move)
            score = -self.quiescence_search(board, -beta, -alpha, depth - 1)
            board.pop()
            
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
                
        return alpha
        
    def _order_moves(self, board, tt_move, ply):
        """着法排序"""
        moves = []
        for move in board.legal_moves:
            score = 0
            
            # 置换表着法
            if tt_move and move == tt_move:
                score = 10000000
                
            # MVV/LVA (Most Valuable Victim / Least Valuable Attacker)
            elif board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim and attacker:
                    score = 1000000 + (victim.piece_type * 10 - attacker.piece_type)
                    
            # 杀手着法
            elif move in self.killer_moves[ply]:
                score = 900000
                
            # 历史启发
            else:
                key = (move.from_square, move.to_square)
                score = self.history_table.get(key, 0)
                
            moves.append((move, score))
            
        # 按分数降序排序
        moves.sort(key=lambda x: x[1], reverse=True)
        return [m[0] for m in moves]
        
    def alpha_beta(self, board, depth, alpha, beta, ply=0, can_null=True):
        """优化的Alpha-Beta搜索"""
        if self.is_time_up():
            return alpha
            
        self.nodes += 1
        
        # 置换表查询
        board_hash = board.fen()
        tt_entry = self.tt.get(board_hash)
        if tt_entry and tt_entry[1] >= depth:
            return tt_entry[0]  # 简化置换表使用
            
        if depth <= 0:
            return self.quiescence_search(board, alpha, beta)
            
        # 简化空着裁剪
        if can_null and depth >= 2 and not board.is_check():
            board.push(chess.Move.null())
            null_value = -self.alpha_beta(board, depth-2, -beta, -beta+1, ply+1, False)
            board.pop()
            if null_value >= beta:
                return beta
                
        moves = self._order_moves(board, None, ply)
        best_value = float('-inf')
        
        for move in moves:
            board.push(move)
            value = -self.alpha_beta(board, depth-1, -beta, -alpha, ply+1)
            board.pop()
            
            if value > best_value:
                best_value = value
                
            alpha = max(alpha, value)
            if alpha >= beta:
                break
                
        # 简化置换表存储
        if len(self.tt) < self.tt_size:
            self.tt[board_hash] = (best_value, depth)
            
        return best_value
        
    def get_best_move(self, board, search_depth=2, time_limit=None):
        """获取最佳着法
        Args:
            board: 棋盘对象
            search_depth: 搜索深度，默认为2
            time_limit: 时间限制（秒）
        """
        if board.fen() in self.opening_book:
            return chess.Move.from_uci(np.random.choice(self.opening_book[board.fen()]))
            
        self.start_time = time.time()
        self.time_limit = time_limit
        self.nodes = 0
        
        # 动态调整搜索深度
        depth = min(search_depth, 3)  # 限制最大深度
            
        best_move = None
        best_value = float('-inf')
        
        moves = self._order_moves(board, None, 0)
        if moves:
            best_move = moves[0]  # 确保有一个默认着法
            
        try:
            for move in moves:
                if self.is_time_up():
                    break
                    
                board.push(move)
                value = -self.alpha_beta(board, depth-1, float('-inf'), float('inf'))
                board.pop()
                
                if value > best_value:
                    best_value = value
                    best_move = move
                    
        except TimeoutError:
            pass
            
        return best_move

def chess_bot(obs, config):
    """优化的代理接口"""
    global engine
    if 'engine' not in globals():
        engine = StockfishNNUE()  # 确保使用正确的类名
        
    board = chess.Board(obs.get("board"))
    
    # 更积极的时间管理
    if "remainingTime" in obs:
        time_limit = min(obs["remainingTime"] / 1000.0 / 40, 0.1)  # 最多使用0.1秒
    else:
        time_limit = 0.1
        
    try:
        # 确保方法调用与定义匹配
        move = engine.get_best_move(board, 8, time_limit)  # 移除关键字参数
    except Exception as e:
        # 如果出错，至少返回一个合法着法
        legal_moves = list(board.legal_moves)
        if legal_moves:
            return legal_moves[0].uci()
        return None
        
    return move.uci()


def run_multiple_games(n_games=5):
    """运行多局对弈测试"""
    results = []
    last_env = None
    
    # 使用tqdm显示进度
    for game_id in tqdm(range(n_games), desc="对弈进度"):
        try:
            # 创建新的环境
            env = make("chess", debug=True)
            last_env = env  # 保存最后一个环境实例
            
            # 设置环境参数
            # env.configuration.episodeSteps = 1000
            # env.configuration.actTimeout = 30
            # env.configuration.agentTimeout = 180
            
            # 运行对局
            outcome = env.run([chess_bot, "random"])
            results.append(outcome)
            
            # 解析最后一步结果
            final_state = outcome[-1][0]
            print(f"\n对局 {game_id + 1} 结果:")
            print(f"白方得分: {final_state.reward} - 状态: {final_state.status}")
            final_state = outcome[-1][1]
            print(f"黑方得分: {final_state.reward} - 状态: {final_state.status}")
            
        except Exception as e:
            print(f"\n对局 {game_id + 1} 发生错误: {str(e)}")
            continue
            
    return results, last_env

def analyze_results(results):
    """分析对弈结果"""
    white_wins = 0
    black_wins = 0
    draws = 0
    total_moves = 0
    
    for result in results:
        try:
            last_step = result[-1]
            white_reward = last_step[0].reward
            black_reward = last_step[1].reward
            
            if white_reward == 1:
                white_wins += 1
            elif black_reward == 1:
                black_wins += 1
            else:
                draws += 1
                
            # 计算总步数
            total_moves += len(result)
            
        except Exception as e:
            print(f"分析结果时发生错误: {str(e)}")
            continue
            
    n_games = len(results)
    avg_moves = total_moves / n_games if n_games > 0 else 0
    
    print("\n=== 最终统计 ===")
    print(f"总对局数: {n_games}")
    print(f"白方胜率: {white_wins}/{n_games} ({white_wins/n_games*100:.1f}%)")
    print(f"黑方胜率: {black_wins}/{n_games} ({black_wins/n_games*100:.1f}%)")
    print(f"平局率: {draws}/{n_games} ({draws/n_games*100:.1f}%)")
    print(f"平均每局步数: {avg_moves:.1f}")
    
    return {
        'white_wins': white_wins,
        'black_wins': black_wins,
        'draws': draws,
        'avg_moves': avg_moves
    }


from tqdm import tqdm

# 运行5次对局并显示统计结果
n_games = 5
results, last_env = run_multiple_games(n_games)

# 分析结果
stats = analyze_results(results)

# 渲染最后一局
if last_env is not None:
    try:
        last_env.render(mode="ipython", width=600, height=600)
    except Exception as e:
        print(f"\n渲染棋盘时发生错误: {str(e)}")
        
# 保存引擎实例的评估数据
if 'engine' in globals():
    print("\n=== 引擎统计 ===")
    print(f"置换表大小: {len(engine.tt)}")
    print(f"历史启发表大小: {len(engine.history_table)}")
    print(f"NNUE状态: {'已加载' if engine.nnue is not None else '使用基础评估'}") 



%%writefile main.py
import os
import time
import sys

# 自动安装 chess 库并延迟执行
def install_and_wait():
    print("正在安装 chess 库...")
    os.system('pip install chess')  # 安装 chess 库
    time.sleep(5)  # 等待 5 秒，确保安装完成
    print("chess 库安装完成，继续执行脚本...")

# 检查是否安装了 chess 库，如果没有则安装
try:
    import chess
except ImportError:
    install_and_wait()
    import chess  # 重新尝试导入
    
import numpy as np
from ctypes import cdll, c_char_p
import os
import time
from kaggle_environments import make

class StockfishNNUE:
    """Stockfish NNUE评估器"""
    def __init__(self, nnue_path="/kaggle/input/nnue/other/default/1/nn-c3ca321c51c9.nnue"):
        # 加载NNUE动态库
        try:
            self.nnue = cdll.LoadLibrary("/kaggle/input/nnue/other/default/1/libnnueprobe.so")
            self.nnue.nnue_init(nnue_path.encode('utf-8'))
        except:
            print("警告: 无法加载NNUE库，将使用基础评估函数")
            self.nnue = None
            
        # 增加置换表大小限制
        self.tt_size = 100000  # 减小置换表大小以加快访问速度
        self.tt = {}
        
        # 增加时间管理参数
        self.nodes = 0
        self.start_time = 0
        self.time_limit = 0
        
        # 历史启发表
        self.history_table = {}
        
        # 杀手着法表
        self.killer_moves = [[None, None] for _ in range(100)]
        
        # 开局库
        self.opening_book = self._init_opening_book()
        
    def _init_opening_book(self):
        """初始化一个简单的开局库"""
        book = {
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1": [
                "e2e4", "d2d4", "c2c4"  # 主要的开局着法
            ]
        }
        return book
        
    def evaluate_position(self, board):
        """评估局面分数"""
        # 终局评估
        if board.is_checkmate():
            return -10000 if board.turn else 10000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
            
        # 使用NNUE评估
        if self.nnue is not None:
            try:
                return self.nnue.nnue_evaluate_fen(board.fen().encode('utf-8'))
            except:
                pass
                
        # 后备的基础评估
        return self._basic_evaluate(board)
        
    def _basic_evaluate(self, board):
        """基础局面评估"""
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        # 子力价值
        score = 0
        for piece_type in piece_values:
            score += len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
            score -= len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]
            
        # 位置价值(简化版)
        pawn_table = [
            0,  0,  0,  0,  0,  0,  0,  0,
            50, 50, 50, 50, 50, 50, 50, 50,
            10, 10, 20, 30, 30, 20, 10, 10,
            5,  5, 10, 25, 25, 10,  5,  5,
            0,  0,  0, 20, 20,  0,  0,  0,
            5, -5,-10,  0,  0,-10, -5,  5,
            5, 10, 10,-20,-20, 10, 10,  5,
            0,  0,  0,  0,  0,  0,  0,  0
        ]
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                if piece.piece_type == chess.PAWN:
                    if piece.color:
                        score += pawn_table[square]
                    else:
                        score -= pawn_table[63 - square]
                        
        return score
        
    def is_time_up(self):
        """检查是否超时"""
        return self.time_limit and (time.time() - self.start_time) > self.time_limit
        
    def quiescence_search(self, board, alpha, beta, depth=0):
        """简化的静态搜索"""
        if self.is_time_up():
            return alpha
            
        stand_pat = self.evaluate_position(board)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
            
        if depth <= -2:  # 减少静态搜索深度
            return stand_pat
            
        # 只考虑有价值的吃子着法
        for move in board.legal_moves:
            if not board.is_capture(move):
                continue
                
            # 快速评估吃子价值
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            if victim and attacker and victim.piece_type <= attacker.piece_type:
                continue
                
            board.push(move)
            score = -self.quiescence_search(board, -beta, -alpha, depth - 1)
            board.pop()
            
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
                
        return alpha
        
    def _order_moves(self, board, tt_move, ply):
        """着法排序"""
        moves = []
        for move in board.legal_moves:
            score = 0
            
            # 置换表着法
            if tt_move and move == tt_move:
                score = 10000000
                
            # MVV/LVA (Most Valuable Victim / Least Valuable Attacker)
            elif board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim and attacker:
                    score = 1000000 + (victim.piece_type * 10 - attacker.piece_type)
                    
            # 杀手着法
            elif move in self.killer_moves[ply]:
                score = 900000
                
            # 历史启发
            else:
                key = (move.from_square, move.to_square)
                score = self.history_table.get(key, 0)
                
            moves.append((move, score))
            
        # 按分数降序排序
        moves.sort(key=lambda x: x[1], reverse=True)
        return [m[0] for m in moves]
        
    def alpha_beta(self, board, depth, alpha, beta, ply=0, can_null=True):
        """优化的Alpha-Beta搜索"""
        if self.is_time_up():
            return alpha
            
        self.nodes += 1
        
        # 置换表查询
        board_hash = board.fen()
        tt_entry = self.tt.get(board_hash)
        if tt_entry and tt_entry[1] >= depth:
            return tt_entry[0]  # 简化置换表使用
            
        if depth <= 0:
            return self.quiescence_search(board, alpha, beta)
            
        # 简化空着裁剪
        if can_null and depth >= 2 and not board.is_check():
            board.push(chess.Move.null())
            null_value = -self.alpha_beta(board, depth-2, -beta, -beta+1, ply+1, False)
            board.pop()
            if null_value >= beta:
                return beta
                
        moves = self._order_moves(board, None, ply)
        best_value = float('-inf')
        
        for move in moves:
            board.push(move)
            value = -self.alpha_beta(board, depth-1, -beta, -alpha, ply+1)
            board.pop()
            
            if value > best_value:
                best_value = value
                
            alpha = max(alpha, value)
            if alpha >= beta:
                break
                
        # 简化置换表存储
        if len(self.tt) < self.tt_size:
            self.tt[board_hash] = (best_value, depth)
            
        return best_value
        
    def get_best_move(self, board, search_depth=2, time_limit=None):
        """获取最佳着法
        Args:
            board: 棋盘对象
            search_depth: 搜索深度，默认为2
            time_limit: 时间限制（秒）
        """
        if board.fen() in self.opening_book:
            return chess.Move.from_uci(np.random.choice(self.opening_book[board.fen()]))
            
        self.start_time = time.time()
        self.time_limit = time_limit
        self.nodes = 0
        
        # 动态调整搜索深度
        depth = min(search_depth, 3)  # 限制最大深度
            
        best_move = None
        best_value = float('-inf')
        
        moves = self._order_moves(board, None, 0)
        if moves:
            best_move = moves[0]  # 确保有一个默认着法
            
        try:
            for move in moves:
                if self.is_time_up():
                    break
                    
                board.push(move)
                value = -self.alpha_beta(board, depth-1, float('-inf'), float('inf'))
                board.pop()
                
                if value > best_value:
                    best_value = value
                    best_move = move
                    
        except TimeoutError:
            pass
            
        return best_move

def chess_bot(obs, config):
    """优化的代理接口"""
    global engine
    if 'engine' not in globals():
        engine = StockfishNNUE()  # 确保使用正确的类名
        
    board = chess.Board(obs.get("board"))
    
    # 更积极的时间管理
    if "remainingTime" in obs:
        time_limit = min(obs["remainingTime"] / 1000.0 / 40, 0.1)  # 最多使用0.1秒
    else:
        time_limit = 0.1
        
    try:
        # 确保方法调用与定义匹配
        move = engine.get_best_move(board, 11, time_limit)  # 移除关键字参数
    except Exception as e:
        # 如果出错，至少返回一个合法着法
        legal_moves = list(board.legal_moves)
        if legal_moves:
            return legal_moves[0].uci()
        return None
        
    return move.uci()


from kaggle_environments import make

env = make("chess", debug=True)
# 设置环境参数
env.configuration.episodeSteps = 1000
env.configuration.actTimeout = 30
env.configuration.agentTimeout = 180
result = env.run(["main.py", "random"])
print("Agent exit status/reward/time left: ")
# look at the generated replay.json and print out the agent info
for agent in result[-1]:
    print("\t", agent.status, "/", agent.reward, "/", agent.observation.remainingOverageTime)
print("\n")
# render the game
env.render(mode="ipython", width=1000, height=1000) 

