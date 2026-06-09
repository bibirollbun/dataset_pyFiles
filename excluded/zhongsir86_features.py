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
import warnings

# 忽略警告信息
warnings.filterwarnings('ignore')

# --- 设置绘图环境 ---
try:
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['Noto Sans SC'] 
    plt.rcParams['axes.unicode_minus'] = False
    print("中文字体 'Noto Sans SC' 设置成功。")
except:
    print("警告：未找到指定的 'Noto Sans SC' 字体。部分中文字符可能无法显示。")

def create_agent_features(df_in):
    """
    分解 agent 字符串，为 agent1 和 agent2 创建组件特征及对决特征。
    
    参数:
    df_in (pd.DataFrame): 包含 'agent1' 和 'agent2' 列的 DataFrame。
    
    返回:
    pd.DataFrame: 增加了新特征的 DataFrame。
    """
    df_out = df_in.copy()
    
    # 分解 agent1 和 agent2 的字符串描述
    for i in [1, 2]:
        agent_col = f'agent{i}'
        agent_parts = df_out[agent_col].str.split('-', expand=True)
        
        df_out[f'agent{i}_selection'] = agent_parts[1]
        df_out[f'agent{i}_exploration'] = agent_parts[2].astype(float)
        df_out[f'agent{i}_playout'] = agent_parts[3]
        df_out[f'agent{i}_score_bounds'] = agent_parts[4].astype(bool)

    # 创建对决 (Matchup) 特征
    df_out['is_same_selection'] = (df_out['agent1_selection'] == df_out['agent2_selection']).astype(int)
    df_out['is_same_playout'] = (df_out['agent1_playout'] == df_out['agent2_playout']).astype(int)
    df_out['exploration_diff'] = df_out['agent1_exploration'] - df_out['agent2_exploration']
    
    return df_out

# --- 1. 加载数据 ---
print("正在加载 train.csv...")
df_train = pd.read_csv('/kaggle/input/um-game-playing-strength-of-mcts-variants/train.csv')
print("数据加载完成。")

# --- 2. 特征工程 ---
print("正在创建新特征...")
df_featured = create_agent_features(df_train)
print("特征创建完成。")

# --- 3. 可视化新特征 ---
print("开始生成特征可视化图表...")
sns.set_theme(style="whitegrid")

# 创建 4x2 的子图布局
fig, axes = plt.subplots(4, 2, figsize=(20, 28))
fig.suptitle('Analysis of New Features vs Agent1 Utility', fontsize=24, y=1.02)  # 主标题保持英文

# 图1: agent1 的选择策略 vs 效用值
sns.boxplot(ax=axes[0, 0], data=df_featured, x='agent1_selection', y='utility_agent1')
axes[0, 0].set_title('Agent1 Selection Strategy vs Utility', fontsize=14)  # 英文标题
axes[0, 0].tick_params(axis='x', rotation=25)
axes[0, 0].set_xlabel('Agent1 Selection Strategy', fontsize=12)
axes[0, 0].set_ylabel('Utility of Agent1', fontsize=12)

# 图2: agent2 的选择策略 vs 效用值
sns.boxplot(ax=axes[0, 1], data=df_featured, x='agent2_selection', y='utility_agent1')
axes[0, 1].set_title('Agent2 Selection Strategy vs Utility', fontsize=14)  # 英文标题
axes[0, 1].tick_params(axis='x', rotation=25)
axes[0, 1].set_xlabel('Agent2 Selection Strategy', fontsize=12)
axes[0, 1].set_ylabel('Utility of Agent1', fontsize=12)

# 图3: agent1 的模拟策略 vs 效用值
sns.boxplot(ax=axes[1, 0], data=df_featured, x='agent1_playout', y='utility_agent1')
axes[1, 0].set_title('Agent1 Playout Strategy vs Utility', fontsize=14)  # 英文标题
axes[1, 0].set_xlabel('Agent1 Playout Strategy', fontsize=12)
axes[1, 0].set_ylabel('Utility of Agent1', fontsize=12)

# 图4: agent2 的模拟策略 vs 效用值
sns.boxplot(ax=axes[1, 1], data=df_featured, x='agent2_playout', y='utility_agent1')
axes[1, 1].set_title('Agent2 Playout Strategy vs Utility', fontsize=14)  # 英文标题
axes[1, 1].set_xlabel('Agent2 Playout Strategy', fontsize=12)
axes[1, 1].set_ylabel('Utility of Agent1', fontsize=12)

# 图5: agent1 的分数界定 vs 效用值
sns.boxplot(ax=axes[2, 0], data=df_featured, x='agent1_score_bounds', y='utility_agent1')
axes[2, 0].set_title('Agent1 Score Bounds vs Utility', fontsize=14)  # 英文标题
axes[2, 0].set_xlabel('Agent1 Score Bounds Enabled', fontsize=12)
axes[2, 0].set_ylabel('Utility of Agent1', fontsize=12)

# 图6: agent2 的分数界定 vs 效用值
sns.boxplot(ax=axes[2, 1], data=df_featured, x='agent2_score_bounds', y='utility_agent1')
axes[2, 1].set_title('Agent2 Score Bounds vs Utility', fontsize=14)  # 英文标题
axes[2, 1].set_xlabel('Agent2 Score Bounds Enabled', fontsize=12)
axes[2, 1].set_ylabel('Utility of Agent1', fontsize=12)

# 图7: 探索常数差异 vs 效用值
sns.regplot(ax=axes[3, 0], data=df_featured.sample(n=5000, random_state=1), 
            x='exploration_diff', y='utility_agent1', scatter_kws={'alpha':0.2})
axes[3, 0].set_title('Exploration Constant Difference (A1_exp - A2_exp) vs Utility', fontsize=14)  # 英文标题
axes[3, 0].set_xlabel('Difference in Exploration Constant', fontsize=12)
axes[3, 0].set_ylabel('Utility of Agent1', fontsize=12)

# 图8: 策略是否相同 vs 效用值
df_melted = pd.melt(df_featured, id_vars=['utility_agent1'], 
                    value_vars=['is_same_selection', 'is_same_playout'],
                    var_name='Same_Strategy_Type', value_name='Is_Same')
sns.boxplot(ax=axes[3, 1], data=df_melted, x='Same_Strategy_Type', 
            y='utility_agent1', hue='Is_Same')
axes[3, 1].set_title('Same Strategy vs Utility (0=Different, 1=Same)', fontsize=14)  # 英文标题
axes[3, 1].set_xlabel('Strategy Type', fontsize=12)
axes[3, 1].set_ylabel('Utility of Agent1', fontsize=12)
axes[3, 1].legend(title='Is Same?')

# 设置x轴标签为更友好的名称
axes[3, 1].set_xticklabels(['Selection Strategy', 'Playout Strategy'])

# 调整布局并显示图表
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()

print("可视化图表生成完毕。")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# --- 环境设置 ---
warnings.filterwarnings('ignore')  # 忽略警告信息
sns.set_theme(style="whitegrid", font_scale=1.0)  # 设置seaborn主题

# --- 1. 加载数据 ---
print("正在加载 train.csv 数据...")
try:
    # 尝试从指定路径读取CSV文件
    df_train = pd.read_csv('/kaggle/input/um-game-playing-strength-of-mcts-variants/train.csv')
    print(f"数据加载成功。数据集维度: {df_train.shape}")
except FileNotFoundError:
    print("错误：在指定路径未找到 train.csv 文件")
    df_train = pd.DataFrame()  # 创建空DataFrame作为后备

if not df_train.empty:
    # --- 2. 特征工程 ---
    print("正在创建新特征...")
    
    # 创建总可玩位置特征 = 棋盘可玩位置 + 手牌起始组件
    df_train['TotalPlayableSites'] = (df_train['NumPlayableSitesOnBoard'] + 
                                     df_train['NumStartComponentsHand'])
    
    # 将复杂度分为4个等级（四分位数）
    df_train['ComplexityLevel'] = pd.qcut(
        df_train['TotalPlayableSites'], 
        q=4, 
        labels=['Low', 'Medium', 'High', 'Very High'],  # 保持英文标签（用于可视化）
        duplicates='drop'
    )
    
    # 从agent名称中提取策略类型
    df_train['Selection_Strategy'] = df_train['agent1'].str.split('-').str[1]  # 选择策略
    df_train['Playout_Strategy'] = df_train['agent1'].str.split('-').str[3]    # 模拟策略
    
    print("特征工程完成。")
    
    # --- 3. 可视化分析 ---
    print("正在生成可视化图表...")
    
    # 创建2x2子图布局
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('MCTS Performance Analysis: Game Complexity vs Agent Strategies', 
                 fontsize=16, y=0.98)  # 主标题保持英文
    
    # 图1：复杂度与效用分布（箱线图）
    sns.boxplot(ax=axes[0, 0], data=df_train, x='ComplexityLevel', y='utility_agent1', 
                palette='viridis')
    axes[0, 0].set_title('Agent Performance by Game Complexity')  # 保持英文
    axes[0, 0].set_xlabel('Game Complexity Level')  # 保持英文
    axes[0, 0].set_ylabel('Agent1 Utility Score')   # 保持英文
    
    # 图2：总可玩位置与效用的散点图（抽样3000个点）
    sample_df = df_train.sample(n=min(3000, len(df_train)), random_state=42)
    sns.scatterplot(ax=axes[0, 1], data=sample_df, x='TotalPlayableSites', 
                    y='utility_agent1', hue='ComplexityLevel', alpha=0.6, 
                    palette='Set1')
    axes[0, 1].set_title('Utility vs Total Playable Sites')  # 保持英文
    axes[0, 1].set_xlabel('Total Playable Sites')  # 保持英文
    axes[0, 1].set_ylabel('Agent1 Utility Score')  # 保持英文
    axes[0, 1].legend(title='Complexity')  # 保持英文
    
    # 图3：不同复杂度下选择策略的性能比较
    sns.pointplot(ax=axes[1, 0], data=df_train, x='ComplexityLevel', 
                  y='utility_agent1', hue='Selection_Strategy', 
                  palette='tab10', dodge=0.3)  # dodge参数使点分开显示
    axes[1, 0].set_title('Selection Strategy Performance by Complexity')  # 保持英文
    axes[1, 0].set_xlabel('Game Complexity Level')  # 保持英文
    axes[1, 0].set_ylabel('Mean Utility Score')     # 保持英文
    axes[1, 0].legend(title='Selection Strategy', bbox_to_anchor=(1.05, 1))  # 保持英文
    axes[1, 0].tick_params(axis='x', rotation=45)  # X轴标签旋转45度
    
    # 图4：不同复杂度下模拟策略的性能比较
    sns.pointplot(ax=axes[1, 1], data=df_train, x='ComplexityLevel', 
                  y='utility_agent1', hue='Playout_Strategy', 
                  palette='Set2', dodge=0.3)
    axes[1, 1].set_title('Playout Strategy Performance by Complexity')  # 保持英文
    axes[1, 1].set_xlabel('Game Complexity Level')  # 保持英文
    axes[1, 1].set_ylabel('Mean Utility Score')     # 保持英文
    axes[1, 1].legend(title='Playout Strategy', bbox_to_anchor=(1.05, 1))  # 保持英文
    axes[1, 1].tick_params(axis='x', rotation=45)  # X轴标签旋转45度
    
    plt.tight_layout()  # 自动调整子图间距
    plt.show()
    
    # --- 4. 摘要统计 ---
    print("\n=== 分析总结 ===")
    print(f"分析的游戏总数: {len(df_train):,}")
    print(f"复杂度范围: {df_train['TotalPlayableSites'].min():.0f} - {df_train['TotalPlayableSites'].max():.0f}")
    print(f"平均效用分数: {df_train['utility_agent1'].mean():.3f}")
    
    print("\n策略分布统计:")
    print(f"选择策略类型数量: {df_train['Selection_Strategy'].nunique()}")
    print(f"模拟策略类型数量: {df_train['Playout_Strategy'].nunique()}")
    
else:
    print("由于数据加载失败，分析无法继续。")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

# Set font for plots
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SimplifiedMCTSFeatureEngineering:
    """Simplified MCTS Agent Feature Engineering Class"""
    
    def __init__(self, train_path):
        """Initialize and load data"""
        self.df_train = pd.read_csv(train_path)
        
    def parse_agent_components(self):
        """Parse agent strings to extract MCTS component features"""
        print("=== Parsing Agent Components ===")
        
        # Parse agent1 and agent2
        for agent_col in ['agent1', 'agent2']:
            agent_parts = self.df_train[agent_col].str.split('-', expand=True)
            self.df_train[f'{agent_col}_selection'] = agent_parts[1]
            self.df_train[f'{agent_col}_exploration'] = agent_parts[2].astype(float)
            self.df_train[f'{agent_col}_playout'] = agent_parts[3]
            self.df_train[f'{agent_col}_score_bounds'] = agent_parts[4].map({'true': 1, 'false': 0})
        
        print("Agent components parsed successfully")
    
    def create_interaction_features(self):
        """Create agent interaction features"""
        print("=== Creating Interaction Features ===")
        
        # Same component features
        self.df_train['same_selection'] = (self.df_train['agent1_selection'] == 
                                          self.df_train['agent2_selection']).astype(int)
        self.df_train['same_playout'] = (self.df_train['agent1_playout'] == 
                                        self.df_train['agent2_playout']).astype(int)
        
        # Exploration difference
        self.df_train['exploration_diff'] = (self.df_train['agent1_exploration'] - 
                                            self.df_train['agent2_exploration'])
        
        # Strategy combinations
        self.df_train['agent1_strategy_combo'] = (self.df_train['agent1_selection'] + '_' + 
                                                 self.df_train['agent1_playout'])
        
        print("Interaction features created")
    
    def create_game_features(self):
        """Create game-related features"""
        print("=== Creating Game Features ===")
        
        # Game statistics
        self.df_train['total_games'] = (self.df_train['num_wins_agent1'] + 
                                       self.df_train['num_draws_agent1'] + 
                                       self.df_train['num_losses_agent1'])
        
        self.df_train['draw_rate'] = self.df_train['num_draws_agent1'] / self.df_train['total_games']
        self.df_train['win_rate_agent1'] = self.df_train['num_wins_agent1'] / self.df_train['total_games']
        
        # Game categories
        game_categories = []
        for game_name in self.df_train['GameRulesetName']:
            if 'Chess' in game_name:
                game_categories.append('Chess')
            elif 'Go' in game_name:
                game_categories.append('Go')
            elif 'Checkers' in game_name:
                game_categories.append('Checkers')
            elif any(x in game_name for x in ['Hunt', 'Fox', 'Hare', 'Dogs']):
                game_categories.append('Hunt')
            elif any(x in game_name for x in ['Tic', 'Tac', 'Toe']):
                game_categories.append('TicTacToe')
            else:
                game_categories.append('Other')
        
        self.df_train['game_category'] = game_categories
        print("Game features created")
    
    def create_competitive_balance_features(self):
        """Create competitive balance features"""
        print("=== Creating Competitive Balance Features ===")
        
        # Game result entropy
        def calculate_result_entropy(wins, draws, losses):
            total = wins + draws + losses
            if total == 0:
                return 0
            probs = [wins/total, draws/total, losses/total]
            probs = [p for p in probs if p > 0]
            return entropy(probs, base=2)
        
        self.df_train['result_entropy'] = self.df_train.apply(
            lambda row: calculate_result_entropy(
                row['num_wins_agent1'], 
                row['num_draws_agent1'], 
                row['num_losses_agent1']
            ), axis=1
        )
        
        # Competition intensity
        self.df_train['win_loss_diff'] = abs(self.df_train['num_wins_agent1'] - self.df_train['num_losses_agent1'])
        self.df_train['competition_intensity'] = 1 / (1 + self.df_train['win_loss_diff'] / self.df_train['total_games'])
        
        # Agent dominance
        self.df_train['agent1_dominance'] = (self.df_train['num_wins_agent1'] - self.df_train['num_losses_agent1']) / self.df_train['total_games']
        
        print("Competitive balance features created")
    
    def analyze_feature_importance(self):
        """Analyze feature importance through correlation"""
        print("=== Feature Importance Analysis ===")
        
        numeric_features = [
            'agent1_exploration', 'agent2_exploration', 'exploration_diff', 
            'total_games', 'draw_rate', 'win_rate_agent1',
            'result_entropy', 'competition_intensity', 'agent1_dominance'
        ]
        
        correlations = {}
        for feature in numeric_features:
            if feature in self.df_train.columns:
                corr = self.df_train[feature].corr(self.df_train['utility_agent1'])
                correlations[feature] = corr
        
        print("Feature correlations with utility_agent1:")
        for feature, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
            print(f"  {feature}: {corr:.4f}")
        
        return correlations
    
    def visualize_features(self):
        """Create 5 key visualizations"""
        print("=== Creating Visualizations ===")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('MCTS Agent Feature Analysis', fontsize=16, fontweight='bold')
        
        # 1. Selection Strategy Performance
        sns.boxplot(data=self.df_train, x='agent1_selection', y='utility_agent1', ax=axes[0,0])
        axes[0,0].set_title('Utility Distribution by Selection Strategy')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Exploration Parameter Impact
        sns.scatterplot(data=self.df_train, x='exploration_diff', y='utility_agent1', 
                       alpha=0.6, ax=axes[0,1])
        axes[0,1].set_title('Exploration Difference vs Utility')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Game Category Performance
        sns.boxplot(data=self.df_train, x='game_category', y='utility_agent1', ax=axes[0,2])
        axes[0,2].set_title('Utility Distribution by Game Category')
        axes[0,2].tick_params(axis='x', rotation=45)
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Competition Intensity vs Utility
        sns.scatterplot(data=self.df_train, x='competition_intensity', y='utility_agent1', 
                       alpha=0.6, color='coral', ax=axes[1,0])
        axes[1,0].set_title('Competition Intensity vs Utility')
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Feature Correlation Heatmap
        feature_cols = ['agent1_exploration', 'agent2_exploration', 'exploration_diff',
                       'draw_rate', 'result_entropy', 'competition_intensity', 
                       'agent1_dominance', 'utility_agent1']
        
        corr_matrix = self.df_train[feature_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                   center=0, ax=axes[1,1])
        axes[1,1].set_title('Feature Correlation Matrix')
        
        # Remove empty subplot
        fig.delaxes(axes[1,2])
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print("\nKey Statistics:")
        print(f"Total games range: [{self.df_train['total_games'].min()}, {self.df_train['total_games'].max()}]")
        print(f"Draw rate range: [{self.df_train['draw_rate'].min():.3f}, {self.df_train['draw_rate'].max():.3f}]")
        print(f"Competition intensity range: [{self.df_train['competition_intensity'].min():.3f}, {self.df_train['competition_intensity'].max():.3f}]")
        print(f"Result entropy range: [{self.df_train['result_entropy'].min():.3f}, {self.df_train['result_entropy'].max():.3f}]")
    
    def get_processed_data(self):
        """Return processed data"""
        return self.df_train

# Example usage for Kaggle
if __name__ == '__main__':
    import os
    
    # 在Kaggle中，数据通常在 /kaggle/input/ 目录下
    print("检查Kaggle输入目录:")
    
    # 方法1: 列出所有可用的数据集
    if os.path.exists('/kaggle/input/'):
        for dataset in os.listdir('/kaggle/input/'):
            print(f"数据集: {dataset}")
            dataset_path = f'/kaggle/input/{dataset}'
            if os.path.isdir(dataset_path):
                print(f"  文件:")
                for file in os.listdir(dataset_path):
                    print(f"    {file}")
    
    # 方法2: 根据原代码中的路径构造
    # 原代码中是: '/kaggle/input/um-game-playing-strength-of-mcts-variants/train.csv'
    data_path = '/kaggle/input/um-game-playing-strength-of-mcts-variants/train.csv'
    
    # 方法3: 如果数据集名字不同，请根据上面的输出修改
    # data_path = '/kaggle/input/你的数据集名字/train.csv'
    
    # 检查文件是否存在
    if os.path.exists(data_path):
        print(f"找到文件: {data_path}")
    else:
        print(f"文件不存在: {data_path}")
        print("请根据上面的输出修改data_path变量")
    
    # Initialize class
    mcts_fe = SimplifiedMCTSFeatureEngineering(data_path)
    
    # Run feature engineering pipeline
    mcts_fe.parse_agent_components()
    mcts_fe.create_interaction_features()
    mcts_fe.create_game_features()
    mcts_fe.create_competitive_balance_features()
    
    # Analyze and visualize
    mcts_fe.analyze_feature_importance()
    mcts_fe.visualize_features()
    
    # Get processed data
    processed_df = mcts_fe.get_processed_data()
    
    print(f"\nProcessed data shape: {processed_df.shape}")
    print("\nNew features created:")
    new_features = [col for col in processed_df.columns if any(x in col for x in 
                   ['_selection', '_exploration', '_playout', 'same_', 'diff', 
                    'combo', 'entropy', 'intensity', 'dominance'])]
    for feature in new_features:
        print(f"  - {feature}")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy  # 用于计算信息熵
import warnings
warnings.filterwarnings('ignore')  # 忽略警告信息

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class SimpleMCTSFeatureEngineer:
    """MCTS特征工程类，用于处理和分析MCTS变体游戏数据"""
    
    def __init__(self, path):
        """初始化方法，加载数据
        
        参数:
        path (str): CSV文件路径
        """
        print(f"正在从 {path} 加载数据...")
        self.df = pd.read_csv(path)
        print(f"数据加载完成，共 {len(self.df)} 行")

    def extract_agent_features(self):
        """从agent字符串中提取特征"""
        print("正在从agent字符串中提取特征...")
        
        # 为agent1和agent2分别提取特征
        for agent in ['agent1', 'agent2']:
            # 将agent字符串按'-'分割成多个部分
            parts = self.df[agent].str.split('-', expand=True)
            
            # 提取各组成部分
            self.df[f'{agent}_selection'] = parts[1]  # 选择策略
            self.df[f'{agent}_exploration'] = parts[2].astype(float)  # 探索常数
            self.df[f'{agent}_playout'] = parts[3]  # 模拟策略
            self.df[f'{agent}_score_bounds'] = parts[4].map({'true': 1, 'false': 0})  # 分数界定
        
        print("Agent特征提取完成")

    def create_basic_features(self):
        """创建基础特征"""
        print("正在创建基础特征...")
        
        # 游戏统计特征
        self.df['total_games'] = self.df[['num_wins_agent1', 'num_draws_agent1', 'num_losses_agent1']].sum(axis=1)
        self.df['draw_rate'] = self.df['num_draws_agent1'] / self.df['total_games']  # 平局率
        self.df['win_rate_agent1'] = self.df['num_wins_agent1'] / self.df['total_games']  # agent1胜率
        
        # 智能体交互特征
        self.df['same_selection'] = (self.df['agent1_selection'] == self.df['agent2_selection']).astype(int)  # 是否相同选择策略
        self.df['exploration_diff'] = self.df['agent1_exploration'] - self.df['agent2_exploration']  # 探索常数差异
        
        # 游戏分类特征
        def classify_game(name):
            """将游戏名称分类到预定义的类别"""
            if 'Chess' in name: return 'Chess'  # 国际象棋类
            if 'Go' in name: return 'Go'  # 围棋类
            if 'Checkers' in name: return 'Checkers'  # 跳棋类
            if any(x in name for x in ['Hunt', 'Fox', 'Hare', 'Dogs']): return 'Hunt'  # 猎人类游戏
            if any(x in name for x in ['Tic', 'Tac', 'Toe']): return 'TicTacToe'  # 井字棋
            return 'Other'  # 其他类型
        
        self.df['game_category'] = self.df['GameRulesetName'].apply(classify_game)  # 应用分类函数
        
        print(f"基础特征创建完成，新增 {self.df.shape[1] - len(self.df.columns)} 个特征")

    def add_competition_features(self):
        """添加竞争相关特征"""
        print("正在添加竞争特征...")
        
        def calc_entropy(row):
            """计算结果分布的信息熵"""
            # 获取胜、平、负的次数
            probs = [row['num_wins_agent1'], row['num_draws_agent1'], row['num_losses_agent1']]
            total = sum(probs)
            if total == 0:
                return 0  # 避免除以零错误
            # 计算概率并过滤零值
            probs = [p / total for p in probs if p > 0]
            # 返回以2为底的信息熵
            return entropy(probs, base=2)
        
        # 结果分布的信息熵（衡量结果不确定性）
        self.df['result_entropy'] = self.df.apply(calc_entropy, axis=1)
        # 胜负差异（绝对值）
        self.df['win_loss_diff'] = abs(self.df['num_wins_agent1'] - self.df['num_losses_agent1'])
        # 竞争激烈程度（差异越小，竞争越激烈）
        self.df['competition_intensity'] = 1 / (1 + self.df['win_loss_diff'] / self.df['total_games'])
        
        print("竞争特征添加完成")

    def visualize_features(self):
        """可视化新创建的特征"""
        print("正在生成特征可视化图表...")
        
        # 创建2x2的子图布局
        plt.figure(figsize=(15, 10))
        
        # 图1: 结果熵分布
        plt.subplot(2, 2, 1)
        sns.histplot(self.df['result_entropy'], bins=30, kde=True)
        plt.title('Distribution of Result Entropy')  # 英文标题
        plt.xlabel('Result Entropy')
        plt.ylabel('Frequency')
        
        # 图2: 竞争激烈程度分布
        plt.subplot(2, 2, 2)
        sns.histplot(self.df['competition_intensity'], bins=30, kde=True)
        plt.title('Distribution of Competition Intensity')  # 英文标题
        plt.xlabel('Competition Intensity')
        plt.ylabel('Frequency')
        
        # 图3: 不同游戏类型的效用值分布
        plt.subplot(2, 2, 3)
        sns.boxplot(data=self.df, x='game_category', y='utility_agent1')
        plt.xticks(rotation=45)  # x轴标签旋转45度避免重叠
        plt.title('Utility Distribution by Game Category')  # 英文标题
        plt.xlabel('Game Category')
        plt.ylabel('Utility of Agent1')
        
        # 图4: 探索差异 vs 效用值
        plt.subplot(2, 2, 4)
        sns.scatterplot(data=self.df, x='exploration_diff', y='utility_agent1', alpha=0.3)
        plt.title('Exploration Difference vs Utility')  # 英文标题
        plt.xlabel('Exploration Constant Difference (Agent1 - Agent2)')
        plt.ylabel('Utility of Agent1')
        
        # 调整布局并显示图表
        plt.tight_layout()
        plt.show()
        
        print("特征可视化完成")

    def get_processed_data(self):
        """获取处理后的数据"""
        return self.df

if __name__ == '__main__':
    # 实例化特征工程类
    engineer = SimpleMCTSFeatureEngineer('/kaggle/input/um-game-playing-strength-of-mcts-variants/train.csv')
    
    # 执行特征工程步骤
    engineer.extract_agent_features()  # 提取agent特征
    engineer.create_basic_features()  # 创建基础特征
    engineer.add_competition_features()  # 添加竞争特征
    
    # 可视化特征
    engineer.visualize_features()

    # 导出结果
    processed_df = engineer.get_processed_data()
    processed_df.to_csv('simple_processed_data.csv', index=False)
    print(f"处理后的数据已保存为 'simple_processed_data.csv'，包含 {processed_df.shape[0]} 行和 {processed_df.shape[1]} 列")

