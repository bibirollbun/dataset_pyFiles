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

