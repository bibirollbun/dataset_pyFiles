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


import os
import sys
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import numpy as np
import polars as pl
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

import kaggle_evaluation.mcts_inference_server


import matplotlib.font_manager as font_manager

# Path to the custom font
font_path = '/kaggle/input/noto-font-dataset/Noto Font Dataset/SimplifiedChinese.ttf'

# Add the custom font to the font manager
font_manager.fontManager.addfont(font_path)

# After adding the font, search for it by filename to get the correct font name
for font in font_manager.fontManager.ttflist:
    if font.fname == font_path:
        print(f"Found font: {font.name}")
        plt.rcParams['font.family'] = font.name
        break





# 读取训练数据集
df_train = pd.read_csv('/kaggle/input/um-game-playing-strength-of-mcts-variants/train.csv')  

# 按照第一个智能体(agent1)分组，计算每个智能体的平均效用值
sta1 = df_train.groupby('agent1', as_index=False)[['utility_agent1']].mean()

# 按照第一个智能体(agent1)分组，统计每个智能体的胜、平、负场次总数
ssta = df_train.groupby('agent1', as_index=False)[['num_wins_agent1','num_draws_agent1','num_losses_agent1']].sum()

# 将胜负平统计数据合并到主统计表中
sta1['num_wins_agent1'] = ssta['num_wins_agent1']      # 胜利场次
sta1['num_draws_agent1'] = ssta['num_draws_agent1']    # 平局场次  
sta1['num_losses_agent1'] = ssta['num_losses_agent1']  # 失败场次

# 计算总比赛场次
sta1['num_total'] = sta1['num_wins_agent1'] + sta1['num_draws_agent1'] + sta1['num_losses_agent1']

# 计算胜率（胜利场次 / 总场次）
sta1['win_rate'] = sta1['num_wins_agent1'] / sta1['num_total']
#sta1['my_utility'] = (sta1['num_wins_agent1']- sta1['num_losses_agent1']) / sta1['num_total']

# 添加计数列，用于后续统计某一范围内的样本数量
sta1['n'] = 1


sta1





import matplotlib.pyplot as plt
import seaborn as sns

# 简单散点图
plt.figure(figsize=(12, 6))
plt.scatter(range(len(sta1)), sta1['utility_agent1'].sort_values(), alpha=0.7)
plt.xlabel('智能体排序')
plt.ylabel('效用值 (utility_agent1)')
plt.title('智能体效用值分布散点图')
plt.grid(True, alpha=0.3)
plt.show()


plt.figure(figsize=(10, 6))
# 箱线图显示分布概况
plt.boxplot(sta1['utility_agent1'], vert=False, widths=0.5)
# 叠加散点图显示每个数据点
plt.scatter(sta1['utility_agent1'], [1]*len(sta1), alpha=0.6, s=50)
plt.xlabel('效用值 (utility_agent1)')
plt.title('智能体效用值分布（箱线图+散点图）')
plt.grid(True, alpha=0.3)
plt.show()


plt.figure(figsize=(10, 6))
sns.kdeplot(data=sta1['utility_agent1'], fill=True, alpha=0.7)
# 叠加实际数据点
plt.scatter(sta1['utility_agent1'], [1]*len(sta1), alpha=0.6, s=30, color='red')
plt.xlabel('效用值 (utility_agent1)')
plt.ylabel('密度')
plt.title('智能体效用值核密度估计')
plt.grid(True, alpha=0.3)
plt.show()


# 如果想按MCTS变体组件分析
sta1['selection'] = sta1['agent1'].str.split('-').str[1]
sta1['exploration'] = sta1['agent1'].str.split('-').str[2]

plt.figure(figsize=(12, 6))
sns.boxplot(data=sta1, x='selection', y='utility_agent1')
plt.xticks(rotation=45)
plt.title('不同选择策略的效用值分布')
plt.show()


gamerule_list = df_train['GameRulesetName'].unique()
len(gamerule_list)


my_list=['Ludus_CoriovalliHaretavl_Four_Dogs_Two_Hares_-_Both_Extension_Joined_Diagonal_Suggested',
 'Ludus_CoriovalliHaretavl_Four_Dogs_Two_Hares_-_Both_Extension_No_Joined_Diagonal_Suggested',
 'Ludus_CoriovalliHaretavl_Four_Dogs_Two_Hares_-_No_Extension_Joined_Diagonal_Suggested',
 'Ludus_CoriovalliHaretavl_Four_Dogs_Two_Hares_-_No_Extension_No_Joined_Diagonal_Suggested',
 'Ludus_CoriovalliHaretavl_Four_Dogs_Two_Hares_-_Top_Extension_Joined_Diagonal_Suggested',
 'Ludus_CoriovalliHaretavl_Four_Dogs_Two_Hares_-_Top_Extension_No_Joined_Diagonal_Suggested',
 'Ludus_CoriovalliHaretavl_Four_Dogs_Two_Hares_Starting_Position_1_-_Both_Extensions_Joined_Diagonal_Suggested',
 'Ludus_CoriovalliHaretavl_Four_Dogs_Two_Hares_Starting_Position_1_-_Both_Extensions_No_Joined_Diagonal_Suggested',
 'Ludus_CoriovalliHaretavl_Four_Dogs_Two_Hares_Starting_Position_1_-_No_Extension_Joined_Diagonal_Suggested',
 'Ludus_CoriovalliHaretavl_Four_Dogs_Two_Hares_Starting_Position_1_-_No_Extension_No_Joined_Diagonal_Suggested',
 'Ludus_CoriovalliHaretavl_Four_Dogs_Two_Hares_Starting_Position_1_-_Top_Extension_Joined_Diagonal_Suggested',
 'Ludus_CoriovalliHaretavl_Four_Dogs_Two_Hares_Starting_Position_1_-_Top_Extension_No_Joined_Diagonal_Suggested',
 'Ludus_CoriovalliHaretavl_Four_Dogs_Two_Hares_Starting_Position_2_-_Both_Extensions_Joined_Diagonal_Suggested',
 'Ludus_CoriovalliHaretavl_Four_Dogs_Two_Hares_Starting_Position_2_-_Both_Extensions_No_Joined_Diagonal_Suggested',
 'Ludus_CoriovalliHaretavl_Four_Dogs_Two_Hares_Starting_Position_2_-_No_Extension_Joined_Diagonal_Suggested',
 'Ludus_CoriovalliHaretavl_Four_Dogs_Two_Hares_Starting_Position_2_-_No_Extension_No_Joined_Diagonal_Suggested',
 'Ludus_CoriovalliHaretavl_Four_Dogs_Two_Hares_Starting_Position_2_-_Top_Extension_Joined_Diagonal_Suggested',
 'Ludus_CoriovalliHaretavl_Four_Dogs_Two_Hares_Starting_Position_2_-_Top_Extension_No_Joined_Diagonal_Suggested',
 'Ludus_CoriovalliHaretavl_Four_Dogs_Two_Hares_Switch_Starting_Position_1_-_Both_Extensions_Joined_Diagonal_Suggested',
 'Ludus_CoriovalliHaretavl_Four_Dogs_Two_Hares_Switch_Starting_Position_1_-_Both_Extensions_No_Joined_Diagonal_Suggested',
 'Ludus_CoriovalliHaretavl_Four_Dogs_Two_Hares_Switch_Starting_Position_1_-_No_Extension_Joined_Diagonal_Suggested',
 'Ludus_CoriovalliHaretavl_Four_Dogs_Two_Hares_Switch_Starting_Position_1_-_No_Extension_No_Joined_Diagonal_Suggested',
 'Ludus_CoriovalliHaretavl_Four_Dogs_Two_Hares_Switch_Starting_Position_1_-_Top_Extension_Joined_Diagonal_Suggested',
 'Ludus_CoriovalliHaretavl_Four_Dogs_Two_Hares_Switch_Starting_Position_1_-_Top_Extension_No_Joined_Diagonal_Suggested',
 'Ludus_CoriovalliHaretavl_Four_Dogs_Two_Hares_Switch_Starting_Position_2_-_Both_Extensions_Joined_Diagonal_Suggested',
 'Ludus_CoriovalliHaretavl_Four_Dogs_Two_Hares_Switch_Starting_Position_2_-_Both_Extensions_No_Joined_Diagonal_Suggested',
 'Ludus_CoriovalliHaretavl_Four_Dogs_Two_Hares_Switch_Starting_Position_2_-_No_Extension_Joined_Diagonal_Suggested',
 'Ludus_CoriovalliHaretavl_Four_Dogs_Two_Hares_Switch_Starting_Position_2_-_No_Extension_No_Joined_Diagonal_Suggested',
 'Ludus_CoriovalliHaretavl_Four_Dogs_Two_Hares_Switch_Starting_Position_2_-_Top_Extension_Joined_Diagonal_Suggested',
 'Ludus_CoriovalliHaretavl_Four_Dogs_Two_Hares_Switch_Starting_Position_2_-_Top_Extension_No_Joined_Diagonal_Suggested']
df_train_filtered =df_train[df_train['GameRulesetName'].isin(my_list)]


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle

def compare_agent_data(df_original, df_filtered, original_name="原始数据", filtered_name="过滤后数据"):
    """
    对比两个数据集中智能体效用值的分布情况
    
    参数:
    df_original: 原始数据集 (包含sta1统计结果)
    df_filtered: 过滤后的数据集 (需要重新计算统计)
    original_name: 原始数据集的标签名称
    filtered_name: 过滤数据集的标签名称
    """
    
    # 如果过滤后的数据需要重新统计，先进行统计
    if 'utility_agent1' not in df_filtered.columns or len(df_filtered) == 0:
        print("过滤后数据为空或缺少utility_agent1列")
        return
    
    # 重新计算过滤后数据的智能体统计
    sta_filtered = df_filtered.groupby('agent1', as_index=False)[['utility_agent1']].mean()
    ssta_filtered = df_filtered.groupby('agent1', as_index=False)[['num_wins_agent1','num_draws_agent1','num_losses_agent1']].sum()
    
    sta_filtered['num_wins_agent1'] = ssta_filtered['num_wins_agent1']
    sta_filtered['num_draws_agent1'] = ssta_filtered['num_draws_agent1'] 
    sta_filtered['num_losses_agent1'] = ssta_filtered['num_losses_agent1']
    sta_filtered['num_total'] = sta_filtered['num_wins_agent1'] + sta_filtered['num_draws_agent1'] + sta_filtered['num_losses_agent1']
    sta_filtered['win_rate'] = sta_filtered['num_wins_agent1'] / sta_filtered['num_total']
    
    # 创建2x2的子图布局
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'MCTS智能体效用值分布对比分析\n{original_name} vs {filtered_name}', fontsize=16, fontweight='bold')
    
    # 1. 散点图对比
    ax1 = axes[0, 0]
    
    # 原始数据散点图
    sorted_orig = df_original['utility_agent1'].sort_values().reset_index(drop=True)
    ax1.scatter(range(len(sorted_orig)), sorted_orig, alpha=0.7, s=50, 
               color='blue', label=f'{original_name} (n={len(sorted_orig)})')
    
    # 过滤数据散点图
    sorted_filt = sta_filtered['utility_agent1'].sort_values().reset_index(drop=True)
    ax1.scatter(range(len(sorted_filt)), sorted_filt, alpha=0.7, s=50, 
               color='red', label=f'{filtered_name} (n={len(sorted_filt)})')
    
    ax1.set_xlabel('智能体排序')
    ax1.set_ylabel('效用值 (utility_agent1)')
    ax1.set_title('1. 散点图对比')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. 箱线图 + 散点图组合
    ax2 = axes[0, 1]
    
    # 准备数据
    box_data = [df_original['utility_agent1'], sta_filtered['utility_agent1']]
    box_labels = [original_name, filtered_name]
    
    # 绘制箱线图
    bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.6)
    
    # 设置箱线图颜色
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # 叠加散点图
    np.random.seed(42)  # 固定随机种子确保可重复性
    for i, data in enumerate(box_data):
        x_scatter = np.random.normal(i+1, 0.04, size=len(data))
        ax2.scatter(x_scatter, data, alpha=0.6, s=30, color=colors[i].replace('light', ''))
    
    ax2.set_ylabel('效用值 (utility_agent1)')
    ax2.set_title('2. 箱线图 + 散点图对比')
    ax2.grid(True, alpha=0.3)
    
    # 3. 核密度估计对比
    ax3 = axes[1, 0]
    
    # 绘制核密度估计
    sns.kdeplot(data=df_original['utility_agent1'], ax=ax3, fill=True, alpha=0.5, 
                color='blue', label=original_name)
    sns.kdeplot(data=sta_filtered['utility_agent1'], ax=ax3, fill=True, alpha=0.5, 
                color='red', label=filtered_name)
    
    # 叠加实际数据点
    ax3.scatter(df_original['utility_agent1'], np.zeros(len(df_original))+1, 
               alpha=0.6, s=20, color='blue')
    ax3.scatter(sta_filtered['utility_agent1'], np.zeros(len(sta_filtered))+1.5, 
               alpha=0.6, s=20, color='red')
    
    ax3.set_xlabel('效用值 (utility_agent1)')
    ax3.set_ylabel('密度')
    ax3.set_title('3. 核密度估计对比')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. MCTS组件分析对比
    ax4 = axes[1, 1]
    
    # 解析MCTS组件
    def parse_mcts_components(df_stat):
        df_copy = df_stat.copy()
        agent_parts = df_copy['agent1'].str.split('-', expand=True)
        df_copy['selection'] = agent_parts[1]  # 选择策略
        df_copy['exploration'] = agent_parts[2]  # 探索常数
        df_copy['playout'] = agent_parts[3]  # 游戏策略
        df_copy['score_bounds'] = agent_parts[4]  # 得分边界
        return df_copy
    
    orig_parsed = parse_mcts_components(df_original)
    filt_parsed = parse_mcts_components(sta_filtered)
    
    # 按选择策略分组的箱线图
    selection_strategies = list(set(orig_parsed['selection'].unique()) | 
                              set(filt_parsed['selection'].unique()))
    
    orig_data_by_selection = [orig_parsed[orig_parsed['selection']==s]['utility_agent1'].values 
                             for s in selection_strategies]
    filt_data_by_selection = [filt_parsed[filt_parsed['selection']==s]['utility_agent1'].values 
                             for s in selection_strategies]
    
    # 创建分组箱线图
    x_positions = np.arange(len(selection_strategies))
    width = 0.35
    
    bp1 = ax4.boxplot(orig_data_by_selection, positions=x_positions - width/2, 
                     widths=width*0.8, patch_artist=True, 
                     boxprops=dict(facecolor='lightblue', alpha=0.7))
    bp2 = ax4.boxplot(filt_data_by_selection, positions=x_positions + width/2, 
                     widths=width*0.8, patch_artist=True,
                     boxprops=dict(facecolor='lightcoral', alpha=0.7))
    
    ax4.set_xticks(x_positions)
    ax4.set_xticklabels(selection_strategies, rotation=45, ha='right')
    ax4.set_ylabel('效用值 (utility_agent1)')
    ax4.set_title('4. 不同选择策略的效用值分布对比')
    ax4.grid(True, alpha=0.3)
    
    # 添加图例
    legend_elements = [Rectangle((0, 0), 1, 1, facecolor='lightblue', alpha=0.7, label=original_name),
                      Rectangle((0, 0), 1, 1, facecolor='lightcoral', alpha=0.7, label=filtered_name)]
    ax4.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    # 打印统计摘要
    print(f"\n=== 数据统计摘要 ===")
    print(f"{original_name}:")
    print(f"  智能体数量: {len(df_original)}")
    print(f"  效用值范围: [{df_original['utility_agent1'].min():.3f}, {df_original['utility_agent1'].max():.3f}]")
    print(f"  效用值均值: {df_original['utility_agent1'].mean():.3f}")
    print(f"  效用值标准差: {df_original['utility_agent1'].std():.3f}")
    
    print(f"\n{filtered_name}:")
    print(f"  智能体数量: {len(sta_filtered)}")
    print(f"  效用值范围: [{sta_filtered['utility_agent1'].min():.3f}, {sta_filtered['utility_agent1'].max():.3f}]")
    print(f"  效用值均值: {sta_filtered['utility_agent1'].mean():.3f}")
    print(f"  效用值标准差: {sta_filtered['utility_agent1'].std():.3f}")

# 使用示例：
# compare_agent_data(sta1, df_train_filtered, "全部游戏", "Four Dogs Two Hares游戏")


compare_agent_data(sta1, df_train_filtered, "全部游戏", "Four Dogs Two Hares游戏")


df_train_filtered=df_train[df_train['agent1']=='MCTS-ProgressiveHistory-0.1-MAST-false']
sta_one_agent=df_train_filtered.groupby('GameRulesetName')[['utility_agent1']].mean()#计算一个算法在不同规则内的平均胜率


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle

def analyze_single_agent_performance(df_train, agent_name):
    """
    分析单个MCTS算法在不同游戏规则下的表现
    
    参数:
    df_train: 训练数据
    agent_name: 要分析的智能体名称
    """
    
    # 过滤出指定算法的数据
    df_filtered = df_train[df_train['agent1'] == agent_name]
    
    if len(df_filtered) == 0:
        print(f"未找到算法 {agent_name} 的数据")
        return
    
    # 计算该算法在不同规则下的统计信息
    stats_by_game = df_filtered.groupby('GameRulesetName').agg({
        'utility_agent1': ['mean', 'std', 'count'],
        'num_wins_agent1': 'sum',
        'num_draws_agent1': 'sum', 
        'num_losses_agent1': 'sum'
    }).round(3)
    
    # 简化列名
    stats_by_game.columns = ['avg_utility', 'std_utility', 'game_count', 'total_wins', 'total_draws', 'total_losses']
    stats_by_game['total_games'] = stats_by_game['total_wins'] + stats_by_game['total_draws'] + stats_by_game['total_losses']
    stats_by_game['win_rate'] = stats_by_game['total_wins'] / stats_by_game['total_games']
    stats_by_game = stats_by_game.reset_index()
    
    # 创建2x2子图布局
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle(f'算法 {agent_name} 在不同游戏规则下的表现分析', fontsize=16, fontweight='bold')
    
    # 1. 直方图：平均效用值分布
    ax1 = axes[0, 0]
    
    # 使用seaborn的histplot绘制直方图
    sns.histplot(data=stats_by_game, x='avg_utility', bins=30, kde=True, 
                ax=ax1, color='skyblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # 添加统计信息
    mean_utility = stats_by_game['avg_utility'].mean()
    median_utility = stats_by_game['avg_utility'].median()
    
    ax1.axvline(mean_utility, color='red', linestyle='--', linewidth=2, 
               label=f'均值: {mean_utility:.3f}')
    ax1.axvline(median_utility, color='orange', linestyle='--', linewidth=2, 
               label=f'中位数: {median_utility:.3f}')
    ax1.axvline(0, color='black', linestyle='-', alpha=0.7, linewidth=1, 
               label='中性表现线')
    
    ax1.set_xlabel('平均效用值')
    ax1.set_ylabel('游戏规则数量')
    ax1.set_title('1. 平均效用值分布直方图')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 添加统计文本
    ax1.text(0.02, 0.98, f'游戏总数: {len(stats_by_game)}\n'
                         f'标准差: {stats_by_game["avg_utility"].std():.3f}\n'
                         f'最小值: {stats_by_game["avg_utility"].min():.3f}\n'
                         f'最大值: {stats_by_game["avg_utility"].max():.3f}', 
            transform=ax1.transAxes, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. 散点图：效用值 vs 胜率
    ax2 = axes[0, 1]
    
    scatter = ax2.scatter(stats_by_game['avg_utility'], stats_by_game['win_rate'], 
                         s=stats_by_game['total_games']*2, 
                         c=stats_by_game['game_count'], 
                         cmap='viridis', alpha=0.7)
    
    ax2.set_xlabel('平均效用值')
    ax2.set_ylabel('胜率')
    ax2.set_title('2. 效用值 vs 胜率关系图\n(气泡大小=总对局数，颜色=游戏数量)')
    ax2.grid(True, alpha=0.3)
    
    # 添加对角线参考
    ax2.plot([-1, 1], [0, 1], 'r--', alpha=0.5, label='理论线性关系')
    ax2.legend()
    
    # 添加颜色条
    plt.colorbar(scatter, ax=ax2, label='游戏规则数量')
    
    # 3. 箱线图：效用值分布 + 误差条
    ax3 = axes[1, 0]
    
    # 按效用值分组（表现好、中等、差）
    def categorize_performance(utility):
        if utility > 0.3:
            return '表现优秀'
        elif utility > -0.3:
            return '表现中等'
        else:
            return '表现较差'
    
    stats_by_game['performance_cat'] = stats_by_game['avg_utility'].apply(categorize_performance)
    
    # 为每个类别创建箱线图
    performance_data = []
    performance_labels = []
    for cat in ['表现较差', '表现中等', '表现优秀']:
        cat_data = stats_by_game[stats_by_game['performance_cat'] == cat]['avg_utility']
        if len(cat_data) > 0:
            performance_data.append(cat_data)
            performance_labels.append(f'{cat}\n(n={len(cat_data)})')
    
    if performance_data:
        bp = ax3.boxplot(performance_data, labels=performance_labels, patch_artist=True)
        colors = ['lightcoral', 'lightyellow', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax3.set_ylabel('平均效用值')
    ax3.set_title('3. 表现分类箱线图')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.7)
    
    # 4. 热力图：游戏特征分析
    ax4 = axes[1, 1]
    
    # 简化游戏名称以便分析
    def extract_game_features(game_name):
        features = {}
        features['has_extension'] = 'Extension' in game_name
        features['has_joined'] = 'Joined' in game_name
        features['has_diagonal'] = 'Diagonal' in game_name
        features['has_switch'] = 'Switch' in game_name
        features['has_position'] = 'Position' in game_name
        return features
    
    # 为每个游戏提取特征
    feature_analysis = []
    for _, row in stats_by_game.iterrows():
        features = extract_game_features(row['GameRulesetName'])
        features['avg_utility'] = row['avg_utility']
        features['game_name'] = row['GameRulesetName']
        feature_analysis.append(features)
    
    feature_df = pd.DataFrame(feature_analysis)
    
    # 计算特征与表现的相关性
    feature_cols = ['has_extension', 'has_joined', 'has_diagonal', 'has_switch', 'has_position']
    correlation_data = []
    
    for feature in feature_cols:
        with_feature = feature_df[feature_df[feature] == True]['avg_utility'].mean()
        without_feature = feature_df[feature_df[feature] == False]['avg_utility'].mean()
        correlation_data.append([with_feature, without_feature])
    
    correlation_matrix = np.array(correlation_data)
    
    # 绘制热力图
    im = ax4.imshow(correlation_matrix.T, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
    
    ax4.set_xticks(range(len(feature_cols)))
    ax4.set_xticklabels([f.replace('has_', '') for f in feature_cols], rotation=45)
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['有该特征', '无该特征'])
    ax4.set_title('4. 游戏特征与表现相关性热力图')
    
    # 添加数值标签
    for i in range(len(feature_cols)):
        for j in range(2):
            text = ax4.text(i, j, f'{correlation_matrix[i, j]:.3f}', 
                           ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=ax4, label='平均效用值')
    
    plt.tight_layout()
    plt.show()
    
    # 打印详细统计信息
    print(f"\n=== {agent_name} 详细表现统计 ===")
    print(f"总游戏规则数: {len(stats_by_game)}")
    print(f"总对局数: {stats_by_game['total_games'].sum()}")
    print(f"整体平均效用值: {stats_by_game['avg_utility'].mean():.3f}")
    print(f"效用值标准差: {stats_by_game['avg_utility'].std():.3f}")
    print(f"最佳表现游戏: {stats_by_game.loc[stats_by_game['avg_utility'].idxmax(), 'GameRulesetName']}")
    print(f"最差表现游戏: {stats_by_game.loc[stats_by_game['avg_utility'].idxmin(), 'GameRulesetName']}")
    
    # 返回统计数据供进一步分析
    return stats_by_game

# 使用示例
# stats = analyze_single_agent_performance(df_train, 'MCTS-ProgressiveHistory-0.1-MAST-false')


stats = analyze_single_agent_performance(df_train, 'MCTS-ProgressiveHistory-0.1-MAST-false')


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_turns_duration_efficiency(df_train, bins=30, figsize=(16, 10)):
    """
    为TurnsDurationEfficiency变量创建综合可视化分析
    
    参数:
    df_train: 包含数据的DataFrame
    bins: 直方图的分箱数
    figsize: 图形大小
    """
    
    # 创建新变量
    df_train['TurnsDurationEfficiency'] = df_train['DurationActions'] / (df_train['DurationTurnsStdDev'] + 0.01)
    
    # 移除可能的无穷大和NaN值
    df_clean = df_train[np.isfinite(df_train['TurnsDurationEfficiency'])].copy()
    
    # 创建2x2子图布局
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('TurnsDurationEfficiency 变量分析\n(DurationActions / DurationTurnsStdDev)', 
                 fontsize=16, fontweight='bold')
    
    # 1. 基础直方图 + 核密度估计
    ax1 = axes[0, 0]
    
    sns.histplot(data=df_clean, x='TurnsDurationEfficiency', bins=bins, kde=True,
                ax=ax1, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # 添加统计线
    mean_val = df_clean['TurnsDurationEfficiency'].mean()
    median_val = df_clean['TurnsDurationEfficiency'].median()
    
    ax1.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
               label=f'均值: {mean_val:.2f}')
    ax1.axvline(median_val, color='orange', linestyle='--', linewidth=2, 
               label=f'中位数: {median_val:.2f}')
    
    ax1.set_xlabel('TurnsDurationEfficiency')
    ax1.set_ylabel('频数')
    ax1.set_title('1. TurnsDurationEfficiency 分布直方图')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 添加统计信息
    stats_text = (f'样本数: {len(df_clean):,}\n'
                 f'标准差: {df_clean["TurnsDurationEfficiency"].std():.2f}\n'
                 f'偏度: {df_clean["TurnsDurationEfficiency"].skew():.2f}\n'
                 f'峰度: {df_clean["TurnsDurationEfficiency"].kurtosis():.2f}')
    
    ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. 对数尺度直方图（处理可能的极值）
    ax2 = axes[0, 1]
    
    # 只处理正值（对数变换需要）
    positive_data = df_clean[df_clean['TurnsDurationEfficiency'] > 0]['TurnsDurationEfficiency']
    
    if len(positive_data) > 0:
        sns.histplot(data=positive_data, bins=bins, kde=True, ax=ax2, 
                    color='forestgreen', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.set_xscale('log')
        ax2.set_xlabel('TurnsDurationEfficiency (对数尺度)')
        ax2.set_ylabel('频数')
        ax2.set_title('2. 对数尺度分布（仅正值）')
        ax2.grid(True, alpha=0.3)
        
        # 添加统计信息
        log_stats = (f'正值样本: {len(positive_data):,}\n'
                    f'几何均值: {np.exp(np.log(positive_data).mean()):.2f}\n'
                    f'最小值: {positive_data.min():.2f}\n'
                    f'最大值: {positive_data.max():.2f}')
        
        ax2.text(0.98, 0.98, log_stats, transform=ax2.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax2.text(0.5, 0.5, '无正值数据', transform=ax2.transAxes, ha='center', va='center')
        ax2.set_title('2. 对数尺度分布（无正值数据）')
    
    # 3. 箱线图 + 小提琴图组合
    ax3 = axes[1, 0]
    
    # 绘制小提琴图
    violin_parts = ax3.violinplot([df_clean['TurnsDurationEfficiency']], positions=[1], 
                                 widths=0.7, showmeans=True, showmedians=True)
    
    # 设置小提琴图颜色
    for pc in violin_parts['bodies']:
        pc.set_facecolor('lightcoral')
        pc.set_alpha(0.7)
    
    # 叠加箱线图
    bp = ax3.boxplot([df_clean['TurnsDurationEfficiency']], positions=[1], 
                    widths=0.3, patch_artist=True, 
                    boxprops=dict(facecolor='white', alpha=0.8),
                    medianprops=dict(color='red', linewidth=2))
    
    ax3.set_ylabel('TurnsDurationEfficiency')
    ax3.set_title('3. 分布形状分析（小提琴图+箱线图）')
    ax3.set_xticks([1])
    ax3.set_xticklabels(['TurnsDurationEfficiency'])
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 标注异常值
    q1 = df_clean['TurnsDurationEfficiency'].quantile(0.25)
    q3 = df_clean['TurnsDurationEfficiency'].quantile(0.75)
    iqr = q3 - q1
    outliers = df_clean[(df_clean['TurnsDurationEfficiency'] < q1 - 1.5*iqr) | 
                       (df_clean['TurnsDurationEfficiency'] > q3 + 1.5*iqr)]
    
    ax3.text(1.5, ax3.get_ylim()[1]*0.9, f'异常值: {len(outliers)}个', 
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # 4. 与效用值的关系散点图
    ax4 = axes[1, 1]
    
    # 如果数据太多，进行采样以提高绘图性能
    if len(df_clean) > 10000:
        sample_data = df_clean.sample(n=10000, random_state=42)
        title_suffix = f' (随机采样 {10000:,} 个点)'
    else:
        sample_data = df_clean
        title_suffix = f' (全部 {len(df_clean):,} 个点)'
    
    scatter = ax4.scatter(sample_data['TurnsDurationEfficiency'], 
                         sample_data['utility_agent1'],
                         alpha=0.5, s=20, c='purple')
    
    ax4.set_xlabel('TurnsDurationEfficiency')
    ax4.set_ylabel('utility_agent1')
    ax4.set_title('4. TurnsDurationEfficiency vs 智能体效用值' + title_suffix)
    ax4.grid(True, alpha=0.3)
    
    # 计算相关系数
    correlation = sample_data['TurnsDurationEfficiency'].corr(sample_data['utility_agent1'])
    ax4.text(0.02, 0.98, f'相关系数: {correlation:.3f}', 
            transform=ax4.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 添加趋势线
    if not sample_data['TurnsDurationEfficiency'].isna().all():
        z = np.polyfit(sample_data['TurnsDurationEfficiency'].dropna(), 
                      sample_data['utility_agent1'].dropna(), 1)
        p = np.poly1d(z)
        x_trend = np.linspace(sample_data['TurnsDurationEfficiency'].min(), 
                             sample_data['TurnsDurationEfficiency'].max(), 100)
        ax4.plot(x_trend, p(x_trend), 'r--', alpha=0.8, linewidth=2, label='趋势线')
        ax4.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 打印详细统计摘要
    print(f"\n=== TurnsDurationEfficiency 统计摘要 ===")
    print(f"总样本数: {len(df_train):,}")
    print(f"有效样本数: {len(df_clean):,}")
    print(f"缺失/无穷值: {len(df_train) - len(df_clean):,}")
    print(f"\n描述性统计:")
    print(df_clean['TurnsDurationEfficiency'].describe())
    
    print(f"\n分位数信息:")
    for p in [1, 5, 10, 90, 95, 99]:
        val = df_clean['TurnsDurationEfficiency'].quantile(p/100)
        print(f"  {p}%分位数: {val:.3f}")
    
    return df_clean

# 使用示例：
# df_clean = plot_turns_duration_efficiency(df_train, bins=50)


df_clean = plot_turns_duration_efficiency(df_train, bins=50)


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def analyze_area_utility_relationship(df_train, figsize=(18, 12)):
    """
    深度分析游戏板面积(Area)与智能体效用值(utility_agent1)的关系
    
    参数:
    df_train: 包含数据的DataFrame
    figsize: 图形大小
    """
    
    # 创建新变量
    df_train['Area'] = df_train['NumRows'] * df_train['NumColumns']
    
    # 移除NaN值
    df_clean = df_train.dropna(subset=['Area', 'utility_agent1']).copy()
    
    # 创建2x3子图布局
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('游戏板面积 vs 智能体效用值 深度关系分析', fontsize=16, fontweight='bold')
    
    # 1. 基础散点图 + 回归线
    ax1 = axes[0, 0]
    
    # 如果数据点太多，进行采样
    if len(df_clean) > 20000:
        sample_data = df_clean.sample(n=20000, random_state=42)
        title_suffix = f' (采样{20000:,}点)'
    else:
        sample_data = df_clean
        title_suffix = f' (全部{len(df_clean):,}点)'
    
    # 绘制散点图
    scatter = ax1.scatter(sample_data['Area'], sample_data['utility_agent1'], 
                         alpha=0.5, s=20, c='steelblue', edgecolors='none')
    
    # 添加回归线
    z = np.polyfit(sample_data['Area'], sample_data['utility_agent1'], 1)
    p = np.poly1d(z)
    x_reg = np.linspace(sample_data['Area'].min(), sample_data['Area'].max(), 100)
    ax1.plot(x_reg, p(x_reg), 'r-', linewidth=2, label=f'线性回归')
    
    # 添加局部回归线(LOWESS)
    from scipy.signal import savgol_filter
    sorted_data = sample_data.sort_values('Area')
    if len(sorted_data) > 50:
        window_length = min(51, len(sorted_data)//10 if len(sorted_data)//10 % 2 == 1 else len(sorted_data)//10 + 1)
        smoothed = savgol_filter(sorted_data['utility_agent1'], window_length, 3)
        ax1.plot(sorted_data['Area'], smoothed, 'orange', linewidth=2, label='局部平滑')
    
    ax1.set_xlabel('游戏板面积 (Area)')
    ax1.set_ylabel('效用值 (utility_agent1)')
    ax1.set_title('1. 散点图 + 回归分析' + title_suffix)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 计算相关性
    correlation = sample_data['Area'].corr(sample_data['utility_agent1'])
    r2 = r2_score(sample_data['utility_agent1'], p(sample_data['Area']))
    
    ax1.text(0.02, 0.98, f'相关系数: {correlation:.4f}\nR²: {r2:.4f}', 
            transform=ax1.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. 分箱分析 - 面积区间的效用值分布
    ax2 = axes[0, 1]
    
    # 创建面积分箱
    n_bins = 15
    df_clean['area_bins'] = pd.cut(df_clean['Area'], bins=n_bins, labels=False)
    
    # 计算每个分箱的统计信息
    bin_stats = df_clean.groupby('area_bins').agg({
        'Area': ['min', 'max', 'mean'],
        'utility_agent1': ['mean', 'std', 'count']
    }).round(3)
    
    bin_stats.columns = ['area_min', 'area_max', 'area_mean', 'utility_mean', 'utility_std', 'count']
    bin_stats = bin_stats.reset_index().dropna()
    
    # 绘制带误差条的折线图
    ax2.errorbar(bin_stats['area_mean'], bin_stats['utility_mean'], 
                yerr=bin_stats['utility_std'], fmt='o-', capsize=5, 
                color='green', alpha=0.8, markersize=6, linewidth=2)
    
    ax2.set_xlabel('平均面积')
    ax2.set_ylabel('平均效用值')
    ax2.set_title('2. 分箱分析 - 误差条图')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 添加样本数量信息
    for i, row in bin_stats.iterrows():
        if i % 3 == 0:  # 每3个点标注一次，避免过密
            ax2.annotate(f'n={int(row["count"])}', 
                        (row['area_mean'], row['utility_mean']), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.7)
    
    # 3. 热力图 - 2D密度图
    ax3 = axes[0, 2]
    
    # 创建2D直方图
    h = ax3.hist2d(sample_data['Area'], sample_data['utility_agent1'], 
                   bins=[30, 20], cmap='YlOrRd', alpha=0.8)
    
    ax3.set_xlabel('游戏板面积 (Area)')
    ax3.set_ylabel('效用值 (utility_agent1)')
    ax3.set_title('3. 2D密度热力图')
    plt.colorbar(h[3], ax=ax3, label='数据点密度')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.7)
    
    # 4. 面积类别的箱线图
    ax4 = axes[1, 0]
    
    # 定义面积类别
    def categorize_area_detailed(area):
        if area <= 16:
            return '很小\n(≤16)'
        elif area <= 49:
            return '小\n(17-49)'
        elif area <= 100:
            return '中\n(50-100)'
        elif area <= 225:
            return '大\n(101-225)'
        else:
            return '很大\n(>225)'
    
    df_clean['area_category'] = df_clean['Area'].apply(categorize_area_detailed)
    
    # 确保类别顺序
    category_order = ['很小\n(≤16)', '小\n(17-49)', '中\n(50-100)', '大\n(101-225)', '很大\n(>225)']
    category_order = [cat for cat in category_order if cat in df_clean['area_category'].unique()]
    
    sns.boxplot(data=df_clean, x='area_category', y='utility_agent1', 
               order=category_order, ax=ax4, palette='Set2')
    
    ax4.set_xlabel('面积类别')
    ax4.set_ylabel('效用值 (utility_agent1)')
    ax4.set_title('4. 面积类别箱线图')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 添加样本数量
    for i, category in enumerate(category_order):
        count = len(df_clean[df_clean['area_category'] == category])
        ax4.text(i, ax4.get_ylim()[1]*0.9, f'n={count}', 
                ha='center', fontsize=10, fontweight='bold')
    
    # 5. 残差分析
    ax5 = axes[1, 1]
    
    # 计算残差
    y_pred = p(sample_data['Area'])
    residuals = sample_data['utility_agent1'] - y_pred
    
    ax5.scatter(sample_data['Area'], residuals, alpha=0.5, s=20, c='purple')
    ax5.axhline(y=0, color='red', linestyle='-', linewidth=2)
    ax5.set_xlabel('游戏板面积 (Area)')
    ax5.set_ylabel('残差 (实际值 - 预测值)')
    ax5.set_title('5. 回归残差分析')
    ax5.grid(True, alpha=0.3)
    
    # 添加残差统计
    residual_std = residuals.std()
    ax5.axhline(y=2*residual_std, color='orange', linestyle='--', alpha=0.7)
    ax5.axhline(y=-2*residual_std, color='orange', linestyle='--', alpha=0.7)
    ax5.text(0.02, 0.98, f'残差标准差: {residual_std:.4f}', 
            transform=ax5.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 6. 统计显著性检验
    ax6 = axes[1, 2]
    
    # 对每个面积类别进行统计检验
    categories = df_clean['area_category'].unique()
    category_data = [df_clean[df_clean['area_category'] == cat]['utility_agent1'].values 
                    for cat in categories if len(df_clean[df_clean['area_category'] == cat]) > 5]
    
    if len(category_data) > 1:
        # 进行方差分析(ANOVA)
        f_stat, p_value = stats.f_oneway(*category_data)
        
        # 绘制均值比较图
        mean_data = []
        std_data = []
        labels = []
        
        for cat in category_order:
            if cat in df_clean['area_category'].unique():
                cat_values = df_clean[df_clean['area_category'] == cat]['utility_agent1']
                if len(cat_values) > 0:
                    mean_data.append(cat_values.mean())
                    std_data.append(cat_values.std())
                    labels.append(cat)
        
        bars = ax6.bar(range(len(mean_data)), mean_data, yerr=std_data, 
                      capsize=5, alpha=0.7, color='lightcoral', 
                      edgecolor='black', linewidth=1)
        
        ax6.set_xticks(range(len(labels)))
        ax6.set_xticklabels(labels, rotation=45, ha='right')
        ax6.set_ylabel('平均效用值')
        ax6.set_title(f'6. 类别均值比较\nANOVA: F={f_stat:.3f}, p={p_value:.4f}')
        ax6.grid(True, alpha=0.3, axis='y')
        ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # 添加显著性标注
        significance = "显著" if p_value < 0.05 else "不显著"
        ax6.text(0.02, 0.98, f'组间差异: {significance}', 
                transform=ax6.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', 
                         facecolor='lightgreen' if p_value < 0.05 else 'lightcoral', 
                         alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # 详细统计报告
    print(f"\n=== 面积 vs 效用值关系分析报告 ===")
    print(f"样本数量: {len(df_clean):,}")
    print(f"面积范围: {df_clean['Area'].min()} - {df_clean['Area'].max()}")
    print(f"效用值范围: {df_clean['utility_agent1'].min():.4f} - {df_clean['utility_agent1'].max():.4f}")
    
    print(f"\n相关性分析:")
    print(f"皮尔逊相关系数: {correlation:.4f}")
    
    # 计算斯皮尔曼相关系数（非参数）
    spearman_corr, spearman_p = stats.spearmanr(sample_data['Area'], sample_data['utility_agent1'])
    print(f"斯皮尔曼相关系数: {spearman_corr:.4f} (p={spearman_p:.4f})")
    
    print(f"线性回归R²: {r2:.4f}")
    print(f"回归方程: y = {z[0]:.6f} * x + {z[1]:.6f}")
    
    print(f"\n各面积类别统计:")
    category_summary = df_clean.groupby('area_category')['utility_agent1'].agg(['count', 'mean', 'std']).round(4)
    print(category_summary)
    
    if 'f_stat' in locals():
        print(f"\nANOVA检验结果:")
        print(f"F统计量: {f_stat:.4f}")
        print(f"p值: {p_value:.6f}")
        print(f"结论: 不同面积类别间的效用值差异{'显著' if p_value < 0.05 else '不显著'}")
    
    return df_clean

# 使用示例：
# df_clean = analyze_area_utility_relationship(df_train)


df_clean = analyze_area_utility_relationship(df_train)


plt.figure(figsize=(25,5))
sns.barplot(data = df_train, x='Area', y='utility_agent1' )

