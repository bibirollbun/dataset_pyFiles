import numpy as np
import matplotlib.pyplot as plt

# 假设模拟100次MCTS迭代的最大深度数据
iterations = np.arange(1, 101)
max_depth = np.random.randint(5, 20, size=100)  # 假设的最大搜索深度

# 绘制图表
plt.figure(figsize=(10, 6))
plt.plot(iterations, max_depth, label='Max Depth', color='b')
plt.xlabel('Iteration')
plt.ylabel('Max Search Depth')
plt.title('Search Depth Over Iterations')
plt.legend()
plt.grid(True)
plt.show()



from scipy.stats import entropy
import numpy as np
import matplotlib.pyplot as plt

# 假设在一次MCTS中，节点访问次数的分布
node_visits = np.random.randint(1, 10, size=10)  # 10个节点的访问次数

# 计算节点访问次数的熵
node_entropy = entropy(node_visits, base=2)

# 绘制图表
plt.figure(figsize=(10, 6))
plt.bar(range(10), node_visits, color='g', alpha=0.7)
plt.xlabel('Node Index')
plt.ylabel('Visit Count')
plt.title(f'Node Visits Distribution\nEntropy = {node_entropy:.2f}')
plt.grid(True)
plt.show()



import numpy as np
import matplotlib.pyplot as plt

# 假设模拟100次MCTS迭代中新扩展的节点数量
iterations = np.arange(1, 101)
expanded_nodes = np.random.randint(5, 15, size=100)  # 每次扩展的节点数

# 计算扩展率的平均值
avg_expansion_rate = np.mean(expanded_nodes)

# 绘制图表
plt.figure(figsize=(10, 6))
plt.plot(iterations, expanded_nodes, label='Expanded Nodes', color='r')
plt.axhline(avg_expansion_rate, color='k', linestyle='--', label=f'Avg Rate = {avg_expansion_rate:.2f}')
plt.xlabel('Iteration')
plt.ylabel('Number of Expanded Nodes')
plt.title('Expansion Rate Over Iterations')
plt.legend()
plt.grid(True)
plt.show()



import numpy as np
import matplotlib.pyplot as plt

# 假设模拟100次MCTS迭代中不同策略的选择频率
iterations = np.arange(1, 101)
strategies = np.random.choice([1, 2, 3, 4], size=100)  # 假设有4种策略

# 计算每种策略的选择频率
strategy_counts = [np.sum(strategies == i) for i in [1, 2, 3, 4]]

# 计算选择频率的方差
strategy_variance = np.var(strategy_counts)

# 绘制图表
plt.figure(figsize=(10, 6))
plt.bar(['Strategy 1', 'Strategy 2', 'Strategy 3', 'Strategy 4'], strategy_counts, color='purple', alpha=0.7)
plt.ylabel('Frequency')
plt.title(f'Strategy Selection Frequencies\nVariance = {strategy_variance:.2f}')
plt.grid(True)
plt.show()


