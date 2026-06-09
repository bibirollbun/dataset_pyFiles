# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 重新加载数据（确保干净）
df = pd.read_csv('/kaggle/input/um-game-playing-strength-of-mcts-variants/train.csv')

# 提取核心 MCTS 类型：取第二个 '-' 之间的部分
def extract_mcts_type(agent_name):
    if pd.isna(agent_name) or agent_name == 'Unknown':
        return 'Unknown'
    parts = str(agent_name).split('-')
    if len(parts) >= 2:
        return parts[1]  # UCB1, UCB1Tuned, ProgressiveHistory
    else:
        return 'Other'

df['mcts_type'] = df['agent1'].apply(extract_mcts_type)

print("简化后的 MCTS 类型分布：")
print(df['mcts_type'].value_counts())


# 数值列安全处理
for col in ['num_wins_agent1', 'num_draws_agent1', 'num_losses_agent1', 'PlayoutsPerSecond', 'GameTreeComplexity', 'BranchingFactorAverage']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# New Feature 1: 胜率
total = df['num_wins_agent1'] + df['num_draws_agent1'] + df['num_losses_agent1']
df['win_rate'] = np.where(total > 0, df['num_wins_agent1'] / total, 0)

# New Feature 2: 每胜一场的模拟开销（越小越好）
df['playouts_per_win'] = np.where(
    df['num_wins_agent1'] > 0,
    df['PlayoutsPerSecond'] / df['num_wins_agent1'],
    9999  # 无胜局时设为高值
)

# New Feature 3: 游戏复杂度指数
df['complexity_index'] = df['GameTreeComplexity'] + df['BranchingFactorAverage']

# New Feature 4: 随机性指数
df['stochastic_hidden_index'] = (
    df['Stochastic'].astype(int) +
    df['HiddenInformation'].astype(int) +
    df['Dice'].astype(int)
)


plt.figure(figsize=(10, 6))
sns.histplot(df['win_rate'], bins=30, kde=True, color='skyblue')
plt.title('New Feature 1: Win Rate Distribution', fontsize=14)
plt.xlabel('Win Rate')
plt.ylabel('Frequency')
plt.xlim(0, 1)
plt.tight_layout()
plt.show()


# 使用简化后的 mcts_type 做 hue
plt.figure(figsize=(12, 7))
filtered_df = df[df['playouts_per_win'] < 1000]  # 过滤极端值

sns.lmplot(
    data=filtered_df,
    x='PlayoutsPerSecond',
    y='win_rate',
    hue='mcts_type',
    scatter_kws={'alpha': 0.4, 's': 30},
    height=7,
    aspect=1.2
)
plt.title('Win Rate vs Playouts Per Second (with Trend Lines)', fontsize=14)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 7))
plot_df = df[df['complexity_index'] > 0]

# 使用 lmplot 加趋势线（推荐！）
g = sns.lmplot(
    data=plot_df,
    x='complexity_index',
    y='win_rate',
    hue='mcts_type',
    scatter_kws={'alpha': 0.3, 's': 30},
    height=7,
    aspect=1.2,
    logx=True  # ←←← 对数 X 轴
)
g.set_axis_labels("Game Complexity Index", "Win Rate")
g.fig.suptitle('Win Rate vs Game Complexity Index (with Trend Lines)', y=1.02)
plt.tight_layout()
plt.show()


# 确保 x 轴包含 0,1,2,3
plt.figure(figsize=(8, 6))
sns.countplot(
    data=df,
    x='stochastic_hidden_index',
    palette='Set2',
    order=[0, 1, 2, 3]  # ←←← 强制显示所有类别
)
plt.title('New Feature 4: Stochastic & Hidden Info Index Distribution', fontsize=14)
plt.xlabel('Index (0=Fully Deterministic, 3=Max Randomness/Hidden)')
plt.ylabel('Count')
plt.xticks([0, 1, 2, 3])  # 明确标注
plt.tight_layout()
plt.show()

