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


# 学号: 2024423320217_ID, 姓名: 林涣然

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy.stats import entropy
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# 全局设置
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用中文字体
plt.rcParams['axes.unicode_minus'] = False   # 正确显示负号
sns.set_style("whitegrid")  # 设置图形风格

# 读取数据
df_train = pd.read_csv('/kaggle/input/um-game-playing-strength-of-mcts-variants/train.csv')

# =================================================================
# 特征1: 规则决策空间密度 (Rule Decision Space Density)
# =================================================================
def calculate_decision_space(rule_text):
    """通过规则文本计算决策空间密度"""
    if pd.isna(rule_text):
        return 0
    
    # 计算关键字出现的频率
    decision_keywords = ['move', 'action', 'choice', 'select', 'decision']
    move_count = sum(rule_text.lower().count(keyword) for keyword in decision_keywords)
    
    # 计算分支复杂度
    branch_points = rule_text.count('if ') + rule_text.count('else') + rule_text.count('case')
    
    # 计算规则长度与复杂度
    length = len(rule_text)
    
    # 综合决策空间密度指标
    return (0.5 * move_count) + (0.3 * branch_points) + (0.2 * length/1000)

df_train['DecisionSpaceDensity'] = df_train['LudRules'].apply(calculate_decision_space)

# 可视化
plt.figure(figsize=(10, 6))
sns.kdeplot(df_train['DecisionSpaceDensity'], fill=True, color='skyblue')
plt.title('特征1: 规则决策空间密度分布', fontsize=14)
plt.xlabel('决策空间密度', fontsize=12)
plt.ylabel('密度', fontsize=12)
plt.grid(alpha=0.2)
plt.savefig('feature1_decision_space.png', dpi=300)
plt.show()

# =================================================================
# 特征2: 规则语义丰富度 (Rule Semantic Richness)
# =================================================================
def calculate_semantic_richness(rule_text):
    """计算规则文本的语义丰富度"""
    if pd.isna(rule_text):
        return 0
    
    # 文本预处理
    text = re.sub(r'[^\w\s]', '', rule_text.lower())
    words = text.split()
    
    if len(words) < 2:
        return 0
    
    # 计算词汇多样性
    unique_words = len(set(words))
    lexical_diversity = unique_words / len(words)
    
    # 计算关键词密度
    keywords = ['game', 'play', 'move', 'board', 'piece', 'player', 'turn', 'action']
    keyword_density = sum(text.count(kw) for kw in keywords) / len(words)
    
    # 计算信息熵
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
    probs = np.array(list(word_counts.values())) / len(words)
    info_entropy = entropy(probs, base=2)
    
    # 综合丰富度指标
    return 0.4 * lexical_diversity + 0.3 * keyword_density + 0.3 * info_entropy

df_train['SemanticRichness'] = df_train['EnglishRules'].apply(calculate_semantic_richness)

# 可视化
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='SemanticRichness', 
    y='utility_agent1', 
    data=df_train.sample(1000),
    alpha=0.6,
    hue='DecisionSpaceDensity',
    palette='viridis',
    size='DecisionSpaceDensity',
    sizes=(10, 100)
)
plt.title('特征2: 规则语义丰富度 vs 智能体效用', fontsize=14)
plt.xlabel('规则语义丰富度', fontsize=12)
plt.ylabel('智能体效用', fontsize=12)
plt.grid(alpha=0.2)
plt.savefig('feature2_semantic_richness.png', dpi=300)
plt.show()

# =================================================================
# 特征3: 智能体性能稳定性 (Agent Performance Stability)
# =================================================================
def calculate_performance_stability(row):
    """计算智能体性能的稳定性指标"""
    wins = row['num_wins_agent1']
    losses = row['num_losses_agent1']
    draws = row['num_draws_agent1']
    
    total_games = wins + losses + draws
    if total_games == 0:
        return 0
    
    # 计算胜率
    win_rate = wins / total_games
    
    # 计算平局率
    draw_rate = draws / total_games
    
    # 稳定性指标（高胜率、低平局率为稳定）
    stability = win_rate * (1 - 0.5 * draw_rate)
    
    return stability

df_train['PerformanceStability'] = df_train.apply(calculate_performance_stability, axis=1)

# 修复分箱问题：添加微小扰动避免重复值
df_train['PerformanceStability_adj'] = df_train['PerformanceStability'] + np.random.normal(0, 1e-6, len(df_train))

# 可视化
plt.figure(figsize=(10, 6))
sns.boxplot(
    x=pd.qcut(df_train['PerformanceStability_adj'], 5),
    y='utility_agent1',
    data=df_train,
    palette='pastel'
)
plt.title('特征3: 性能稳定性 vs 智能体效用', fontsize=14)
plt.xlabel('性能稳定性分组', fontsize=12)
plt.ylabel('智能体效用', fontsize=12)
plt.xticks(rotation=15)
plt.grid(alpha=0.2)
plt.savefig('feature3_performance_stability.png', dpi=300)
plt.show()

# =================================================================
# 特征4: 策略稳健性指数 (Strategy Robustness Index)
# =================================================================
def calculate_strategy_robustness(row):
    """计算策略的稳健性指标"""
    wins = row['num_wins_agent1']
    losses = row['num_losses_agent1']
    draws = row['num_draws_agent1']
    
    total_games = wins + losses + draws
    if total_games == 0:
        return 0
    
    # 计算胜率
    win_rate = wins / total_games
    
    # 计算损失率
    loss_rate = losses / total_games
    
    # 稳健性指标（高胜率、低损失率为稳健）
    robustness = win_rate * (1 - loss_rate)
    
    return robustness

df_train['StrategyRobustness'] = df_train.apply(calculate_strategy_robustness, axis=1)

# 可视化
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='StrategyRobustness',
    y='utility_agent1',
    data=df_train.sample(1000),
    hue='PerformanceStability',
    size='DecisionSpaceDensity',
    sizes=(10, 200),
    palette='coolwarm',
    alpha=0.7
)
plt.title('特征4: 策略稳健性指数 vs 智能体效用', fontsize=14)
plt.xlabel('策略稳健性指数', fontsize=12)
plt.ylabel('智能体效用', fontsize=12)
plt.grid(alpha=0.2)
plt.savefig('feature4_strategy_robustness.png', dpi=300)
plt.show()

# =================================================================
# 综合特征分析
# =================================================================
# 特征归一化
scaler = MinMaxScaler()
features = ['DecisionSpaceDensity', 'SemanticRichness', 'PerformanceStability', 'StrategyRobustness']
X = scaler.fit_transform(df_train[features])
y = df_train['utility_agent1'].values

# 特征相关性分析
corr_matrix = pd.DataFrame(X, columns=features).corrwith(pd.Series(y))

plt.figure(figsize=(10, 6))
sns.barplot(x=corr_matrix.values, y=corr_matrix.index, palette='Blues_r')
plt.title('特征与智能体效用的相关性', fontsize=14)
plt.xlabel('相关系数', fontsize=12)
plt.ylabel('特征', fontsize=12)
plt.grid(alpha=0.2)
plt.savefig('feature_correlation.png', dpi=300)
plt.show()

# 特征组合分析
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x='StrategyRobustness',
    y='DecisionSpaceDensity',
    data=df_train.sample(1000),
    hue='utility_agent1',
    size='PerformanceStability',
    sizes=(10, 200),
    palette='viridis',
    alpha=0.7
)
plt.title('策略稳健性 vs 规则决策空间密度', fontsize=14)
plt.xlabel('策略稳健性指数', fontsize=12)
plt.ylabel('规则决策空间密度', fontsize=12)
plt.grid(alpha=0.2)
plt.savefig('feature_combination.png', dpi=300)
plt.show()

# 模型验证
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

print(f"模型R²分数: {r2:.4f}")

# 特征重要性
importances = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importances, palette='Greens_r')
plt.title('特征重要性分析', fontsize=14)
plt.xlabel('重要性', fontsize=12)
plt.ylabel('特征', fontsize=12)
plt.grid(alpha=0.2)
plt.savefig('feature_importance.png', dpi=300)
plt.show()

