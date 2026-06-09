%%capture
# ensure we are on the latest version of kaggle-environments
!pip install --upgrade kaggle-environments
!pip install kaggle-environments Chessnut torch


%%capture
# Now let's set up the chess environment!
from kaggle_environments import make
env = make("chess", debug=True)


# Run the game with random policy

env.configuration.episodeSteps = 1000
print("Starting the game...")
result = env.run(["random", "random"])

print("Agent exit status/reward/time left: ")
for agent in result[-1]:
    print("\t", agent.status, "/", agent.reward, "/", agent.observation.remainingOverageTime)

env.render(mode="ipython", width=1000, height=1000)


%%writefile main.py

from Chessnut import Game
import random
import math
import pickle

class MCTSNode:
    """
    Represents a node in the Monte Carlo Tree Search tree.
    """

    def __init__(self, game, parent=None, move=None):
        self.game = game  # Chessnut Game object
        self.parent = parent  # Parent node
        self.move = move  # Move that led to this node
        self.children = []  # Child nodes
        self.wins = 0  # Number of wins in simulations
        self.visits = 0  # Number of visits
        self.untried_moves = self.get_legal_moves()  # List of untried moves

    def get_legal_moves(self):
        """Get legal moves from the current game state."""
        return list(self.game.get_moves())

    def is_fully_expanded(self):
        """Check if all possible moves have been tried."""
        return len(self.untried_moves) == 0

    def best_child(self, exploration_weight=1.4):
        """
        Select the best child node based on UCB1 formula.
        UCB = win_rate + C * sqrt( 2 * ln(parent_visits) / child_visits )
        """
        choices_weights = []
        for child in self.children:
            win_rate = child.wins / (child.visits + 1e-9)
            ucb = win_rate + exploration_weight * math.sqrt(
                2.0 * math.log(self.visits + 1e-9) / (child.visits + 1e-9)
            )
            choices_weights.append(ucb)
        return self.children[choices_weights.index(max(choices_weights))]

    def expand(self):
        """
        Expand a new child node from an untried move.
        """
        move = self.untried_moves.pop()
        new_game = Game(self.game.get_fen())  # Create a new game from the current board state
        new_game.apply_move(move)
        child_node = MCTSNode(new_game, parent=self, move=move)
        self.children.append(child_node)
        return child_node

    @staticmethod
    def is_game_over(game):
        """Check if the game is over: checkmate or stalemate."""
        return game.status in [Game.CHECKMATE, Game.STALEMATE]

    def simulate(self, max_depth=100):
        """
        Simulate a random game from the current node and return the result.
        Limit the simulation depth to avoid infinite loops.

        返回值的含义：
        - 1 表示对当前节点一方的胜利
        - 0.5 表示平局
        - 0 表示对当前节点一方的失败
        """
        sim_game = Game(self.game.get_fen())  # copy current board state
        depth = 0

        while not self.is_game_over(sim_game) and depth < max_depth:
            moves = list(sim_game.get_moves())
            if not moves:
                break
            move = random.choice(moves)
            sim_game.apply_move(move)
            depth += 1

        if self.is_game_over(sim_game):
            if sim_game.status == Game.CHECKMATE:
                last_player = 'b' if sim_game.state.player == 'w' else 'w'
                return 1.0 if last_player == self.game.state.player else 0.0
            else:
                return 0.5
        return 0.5

    def backpropagate(self, result):
        """
        Backpropagate the simulation result up to the root.
        result 对当前节点一方而言：1=胜, 0.5=和, 0=败
        """
        self.visits += 1
        self.wins += result

        if self.parent:
            self.parent.backpropagate(1 - result if result in [0, 1] else 0.5)

def mcts_search(root_game, iterations=1000):
    """
    Perform Monte Carlo Tree Search from the given root game state.
    Return the best move (with the most visits).
    """
    root_node = MCTSNode(root_game)

    for _ in range(iterations):
        node = root_node

        while node.is_fully_expanded() and node.children:
            node = node.best_child()

        if not node.is_fully_expanded():
            node = node.expand()

        result = node.simulate()

        node.backpropagate(result)

    best_child_node = max(root_node.children, key=lambda c: c.visits)
    return best_child_node.move

# Training phase
def train_mcts_model(iterations=10):
    """
    Train an MCTS model using simulated games.
    """
    root_game = Game()
    for _ in range(iterations):
        mcts_search(root_game, iterations=30)

    with open("mcts_model.pkl", "wb") as model_file:
        pickle.dump(root_game, model_file)

# Deployment phase
def mcts_bot(obs):
    """
    Chess bot that uses a pre-trained Monte Carlo Tree Search model to choose a move.
    """
    game = Game(obs.board)
    move = mcts_search(game, iterations=10)
    return move


from kaggle_environments import make
env = make("chess")

print("Starting the game...")
env.configuration.episodeSteps = 10
env.configuration.runTimeout = 600
env.configuration.agentTimeout = 600
env.configuration.actTimeout = 600
result = env.run([mcts_bot, "random"])

print("Agent exit status/reward/time left: ")
for agent in result[-1]:
    print("\t", agent.status, "/", agent.reward, "/", agent.observation.remainingOverageTime)

env.render(mode="ipython", width=1000, height=1000)


import random
import math
from Chessnut import Game

class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state  # Current board state (FEN format)
        self.parent = parent
        self.move = move  # Move leading to this node
        self.children = []  # Child nodes
        self.visits = 0  # Number of visits
        self.value = 0  # Accumulated reward value
        self.untried_moves = None  # To cache legal moves

    def is_fully_expanded(self):
        if self.untried_moves is None:
            game = Game(self.state)
            self.untried_moves = list(game.get_moves())
        return len(self.untried_moves) == 0

    def best_child(self, exploration_weight=1.0):
        """Selects the best child node based on the UCT formula"""
        best_score = -float('inf')
        best_child = None
        for child in self.children:
            exploit = child.value / child.visits
            explore = exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
            score = exploit + explore
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def expand(self):
        """Expands a new child node"""
        if self.untried_moves is None:
            game = Game(self.state)
            self.untried_moves = list(game.get_moves())
            random.shuffle(self.untried_moves)  # Shuffle moves for randomness

        if self.untried_moves:
            move = self.untried_moves.pop()
            game = Game(self.state)
            game.apply_move(move)
            child_node = MCTSNode(game.get_fen(), parent=self, move=move)
            self.children.append(child_node)
            return child_node
        else:
            return None  # No moves to expand

    def update(self, reward):
        """Updates the node's statistics"""
        self.visits += 1
        self.value += reward

def mcts_search(root, max_iterations=1000):
    for _ in range(max_iterations):
        node = root
        # 1. Selection: Traverse down based on UCT until a leaf node
        while node.is_fully_expanded() and node.children:
            node = node.best_child()

        # 2. Expansion: Expand the leaf node
        if not node.is_fully_expanded():
            node = node.expand()

        if node is None:
            continue  # No moves to expand, skip to next iteration

        # 3. Simulation: Simulate the game randomly
        reward = simulate_game(node.state)

        # 4. Backpropagation: Update nodes along the path
        while node:
            node.update(reward)
            reward = -reward  # Switch reward for the opponent
            node = node.parent

    return root.best_child(exploration_weight=0).move

def simulate_game(fen, max_depth=100):
    """Simulates a random game and returns a reward value"""
    game = Game(fen)
    depth = 0
    while not game.status and depth < max_depth:
        moves = list(game.get_moves())
        if not moves:
            break  # No possible moves, game over
        move = random.choice(moves)
        game.apply_move(move)
        depth += 1
    # Determine the reward
    if game.status == Game.CHECKMATE:
        # The player who just moved wins
        return 1
    elif game.status == Game.STALEMATE:
        return 0  # Draw
    else:
        # Game did not reach a terminal state within max_depth
        return 0

def minimax_with_alpha_beta(fen, depth, alpha, beta, maximizing_player):
    """Minimax algorithm with Alpha-Beta pruning"""
    game = Game(fen)

    if depth == 0 or game.status:
        return evaluate_board(game)

    moves = list(game.get_moves())
    if not moves:
        # No moves available
        if game.is_in_check():
            # Checkmate
            return -float('inf') if maximizing_player else float('inf')
        else:
            # Stalemate
            return 0

    if maximizing_player:
        max_eval = -float('inf')
        for move in moves:
            game.apply_move(move)
            eval = minimax_with_alpha_beta(game.get_fen(), depth - 1, alpha, beta, False)
            game.undo_move()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Beta cutoff
        return max_eval
    else:
        min_eval = float('inf')
        for move in moves:
            game.apply_move(move)
            eval = minimax_with_alpha_beta(game.get_fen(), depth - 1, alpha, beta, True)
            game.undo_move()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Alpha cutoff
        return min_eval

def evaluate_board(game):
    """Simple board evaluation function, returning a score for the position"""
    piece_values = {'P':1, 'N':3, 'B':3, 'R':5, 'Q':9, 'K':0,
                    'p':-1, 'n':-3, 'b':-3, 'r':-5, 'q':-9, 'k':0}
    board = game.board._position  # List of pieces
    score = 0
    for piece in board:
        score += piece_values.get(piece, 0)
    return score

class HybridAgent:
    def __init__(self, mcts_iterations=500, minimax_depth=3):
        self.mcts_iterations = mcts_iterations
        self.minimax_depth = minimax_depth

    def select_action(self, fen):
        """Selects a move by combining MCTS and Alpha-Beta pruning"""
        # MCTS Part
        root = MCTSNode(fen)
        mcts_move = mcts_search(root, max_iterations=self.mcts_iterations)

        # Minimax Part
        best_score = -float('inf')
        minimax_move = None
        game = Game(fen)
        for move in game.get_moves():
            game.apply_move(move)
            score = minimax_with_alpha_beta(game.get_fen(), self.minimax_depth - 1, -float('inf'), float('inf'), False)
            game.undo_move()
            if score > best_score:
                best_score = score
                minimax_move = move

        # Combine strategies: Choose the move with the higher evaluation
        if best_score > 0:
            return minimax_move
        else:
            return mcts_move


# make environments
env = make("chess")
env.configuration.episodeSteps = 10
env.configuration.runTimeout = 600
env.configuration.agentTimeout = 600
env.configuration.actTimeout = 600

# Run the game
print("Starting the game...")
result = env.run([chess_bot, "random"])

print("Agent exit status/reward/time left: ")
for agent in result[-1]:
    print("\t", agent.status, "/", agent.reward, "/", agent.observation.remainingOverageTime)

env.render(mode="ipython", width=1000, height=1000)


import random
import math
import os
import torch
import torch.nn as nn
import torch.optim as optim

from Chessnut import Game
from kaggle_environments import make


class DQNNet(nn.Module):
    def __init__(self, input_dim=64*6, hidden_dim=128, move_size=200):
        """
        - input_dim: 同上，对棋盘做简易向量化
        - move_size: 预设最大的动作空间(同AlphaZero示例的简化处理方式)
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # 输出Q(s)对应 move_size 个动作的分数
        self.fc3 = nn.Linear(hidden_dim, move_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # [batch_size, move_size]


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        sample_data = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return sample_data

    def __len__(self):
        return len(self.buffer)

def moves_to_indices(moves, max_move_size):
    """
    仅做示例，将 moves 列表映射到 [0, max_move_size) 的 index。
    实际需要更复杂的"move->index"映射。
    """
    move2idx = {}
    for i, m in enumerate(moves[:max_move_size]):
        move2idx[m] = i
    return move2idx

def select_action_dqn(net, fen, epsilon=0.1):
    """
    epsilon-greedy 策略，从DQN网络中选择动作。
    """
    game = Game(fen)
    moves = list(game.get_moves())
    if not moves:
        return None, 0  # no action

    # 先随机
    if random.random() < epsilon:
        move = random.choice(moves)
        return move, 0

    # 否则用网络
    x = fen_to_input_vector(fen).unsqueeze(0)
    with torch.no_grad():
        q_values = net(x)  # [1, move_size]
    q_values = q_values.squeeze(0)
    move2idx = moves_to_indices(moves, q_values.shape[0])
    best_move = None
    best_q = -9999
    for m in moves:
        idx = move2idx[m]
        val = q_values[idx].item()
        if val > best_q:
            best_q = val
            best_move = m
    return best_move, best_q

def dqn_train_one_episode(net, target_net, buffer, optimizer, gamma=0.9, max_steps=30):
    """
    让 DQN 与随机对手对弈一局，并存储数据到 replay buffer。
    示例：只控制白方由DQN来走，黑方随机。
    """
    game = Game()
    done = False
    steps = 0
    fen = game.get_fen()
    current_player = 'w'

    while not done and steps < max_steps:
        moves = list(game.get_moves())
        if not moves:
            done = True
            # 判断赢/输
            break

        if current_player == 'w':
            # DQN agent
            action, _ = select_action_dqn(net, fen, epsilon=0.2)
        else:
            # Random agent for opponent
            action = random.choice(moves)
        
        if action is None:
            done = True
            break

        # apply
        old_fen = fen
        game.apply_move(action)
        fen = game.get_fen()

        # reward
        reward = 0.0
        if game.status == Game.CHECKMATE:
            # 胜负
            loser = game.state.player
            if loser == current_player:
                reward = -1
            else:
                reward = 1
            done = True
        elif game.status == Game.STALEMATE:
            reward = 0
            done = True

        # 存储
        move2idx_map = moves_to_indices(moves, net.fc3.out_features)
        a_idx = move2idx_map[action]
        buffer.push((old_fen, a_idx, reward, fen, done))

        # 切换玩家
        current_player = 'b' if current_player == 'w' else 'w'
        steps += 1

    # 训练一小步
    batch_size = 16
    sample = buffer.sample(batch_size)
    if not sample:
        return
    # 准备tensor
    old_state_list = []
    action_idx_list = []
    reward_list = []
    next_state_list = []
    done_list = []
    for s,a,r,n,d in sample:
        old_state_list.append(s)
        action_idx_list.append(a)
        reward_list.append(r)
        next_state_list.append(n)
        done_list.append(d)

    old_x = torch.stack([fen_to_input_vector(f) for f in old_state_list], dim=0)
    next_x = torch.stack([fen_to_input_vector(f) for f in next_state_list], dim=0)
    reward_t = torch.FloatTensor(reward_list).unsqueeze(-1)
    action_idx_t = torch.LongTensor(action_idx_list).unsqueeze(-1)
    done_t = torch.BoolTensor(done_list).unsqueeze(-1)

    q_values = net(old_x)  # [B, move_size]
    q_a = q_values.gather(1, action_idx_t)  # [B,1]

    with torch.no_grad():
        # double dqn 或者普通 dqn
        q_next = target_net(next_x)
        max_q_next = q_next.max(dim=1, keepdim=True)[0]  # [B,1]
        target = reward_t + gamma * max_q_next * (~done_t)

    loss_fn = nn.MSELoss()
    loss = loss_fn(q_a, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def dqn_deploy(obs, net):
    """
    在 Kaggle Env 中使用的 DQN 推理入口
    """
    fen = obs.board
    move, _ = select_action_dqn(net, fen, epsilon=0.0)
    return move


def save_model(net, path="model.pth"):
    torch.save(net.state_dict(), path)

def load_model(net, path="model.pth"):
    if os.path.exists(path):
        net.load_state_dict(torch.load(path))
        net.eval()


def fen_to_input_vector(fen):
    """
    将 FEN 转换为简单的向量(示例版)，仅对主要棋子做 one-hot，64*6。
    真实可改用更合理的表征(CNN或embedding)。
    """
    # FEN = "board player castling enpassant halfmove fullmove"
    board_part = fen.split(' ')[0]
    # 先转 64 个字符
    game = Game(fen)
    pos = game.board._position  # 64 长度列表
    # 对每个格子做 one-hot: (P, N, B, R, Q, K) 各一种，黑子的用小写
    # 这里只演示 6类*(白+黑=12种) => 不做区分颜色时可能不精确, 仅作示例
    piece_map = {'P':0, 'N':1, 'B':2, 'R':3, 'Q':4, 'K':5,
                 'p':0, 'n':1, 'b':2, 'r':3, 'q':4, 'k':5}
    out = [0]*(64*6)
    for i, ch in enumerate(pos):
        if ch in piece_map:
            idx = i*6 + piece_map[ch]
            out[idx] = 1
    return torch.FloatTensor(out)

def get_policy_for_moves(net, fen, moves):
    """
    调用策略网络，得到对 moves 的先验概率映射: move -> p
    这里只是演示，不做真正的‘move编码’，随机分配大一些概率。
    """
    x = fen_to_input_vector(fen).unsqueeze(0)  # [1, input_dim]
    with torch.no_grad():
        logit, value = net(x)  # logit: [1, move_size], value: [1,1]
    # 这里实际上应该对"所有可能走法"进行编码，然后只取相应 move 的部分
    # 我们简化为: 用 softmax(logit) 里的前 len(moves) 个通道映射到 moves
    # 真实实现需有“move-> index”映射，这里只做演示
    policy_vec = torch.softmax(logit, dim=1).squeeze(0)  # [move_size]
    # 取前 len(moves) 的值(或散列取)
    portion = min(len(moves), len(policy_vec))
    vals = policy_vec[:portion].tolist()
    s = sum(vals) + 1e-8
    move2p = {}
    for i, m in enumerate(moves[:portion]):
        move2p[m] = vals[i]/s
    # 对超出 portion 范围的 moves 给个小概率
    for m in moves[portion:]:
        move2p[m] = 0.001
    return move2p


# 1) 初始化网络 & target_net
net = DQNNet()
target_net = DQNNet()
target_net.load_state_dict(net.state_dict())
optimizer = optim.Adam(net.parameters(), lr=1e-3)

# 2) replay buffer
buffer = ReplayBuffer()

# 3) 训练若干回合
print("DQN training start...")
for episode in range(100):
    dqn_train_one_episode(net, target_net, buffer, optimizer, gamma=0.9, max_steps=15)
    # 定期更新 target_net
    target_net.load_state_dict(net.state_dict())
print("DQN training done.")

# 4) 保存
save_model(net, "dqn_demo.pth")


def evaluate_agent(env, agent1, agent2, n_games=10):
    """
    让 agent1 与 agent2 下 n_games 局，打印胜平负统计。
    注意：Kaggle Env 自带“随机开局”设定，可能需要关掉或做相应处理。
    """
    wins, draws, losses = 0, 0, 0
    for _ in range(n_games):
        # 运行对局
        result = env.run([agent1, agent2])
        # 取最后一步的 obs
        final_info = result[-1]
        # final_info 是个列表 [info_for_agent1, info_for_agent2]
        # 里边包含 status, reward 等
        r1 = final_info[0].reward
        r2 = final_info[1].reward
        if r1 > r2:
            wins += 1
        elif r1 == r2:
            draws += 1
        else:
            losses += 1
        env.render(mode="ipython", width=1000, height=1000)
    print(f"Agent1 vs Agent2: W/D/L = {wins}/{draws}/{losses}")


env = make("chess", configuration={"episodeSteps":1000, "randomOpenings":False})

def dqn_agent(obs):
    return dqn_deploy(obs, net)
evaluate_agent(env, dqn_agent, "random", n_games=2)

