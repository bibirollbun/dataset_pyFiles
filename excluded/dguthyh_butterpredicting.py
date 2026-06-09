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

# 定义文件路径和所有模型名称
path = '/kaggle/input/19-june-2025-fertilizer/submission__LB__0_'
st_names = ['37_474','37_631','37_780','37_856','GEN_1','GEN_2','GEN_3','GEN_4','GEN_5']

# 读取所有提交文件
print("Loading submission files...")
dfs = [pd.read_csv(path + name_subm + '.csv') for name_subm in st_names]

# 重命名列以便区分
print("Renaming columns...")
for i, df in enumerate(dfs):
    dfs[i] = dfs[i].rename(columns={'Fertilizer Name': f'model_{i}'})

# 合并所有提交
print("Merging data frames...")
merged = dfs[0][['id', 'model_0']]
for i in range(1, len(dfs)):
    merged = merged.merge(dfs[i][['id', f'model_{i}']], on='id')

# 创建优化的集成预测
def ensemble_predict(row):
    # 提取前4个模型的预测
    top4 = [row[f'model_{i}'] for i in range(4)]
    
    # 检查是否存在3个一致且不同于第4个的情况
    for i in range(4):
        # 检查其他三个是否相同
        others = [val for j, val in enumerate(top4) if j != i]
        if len(set(others)) == 1 and others[0] != top4[i]:
            return others[0]
    
    # 如果没有找到3:1的情况，使用表现最佳的GEN_5模型
    return row['model_8']

print("Applying ensemble strategy...")
merged['Fertilizer Name'] = merged.apply(ensemble_predict, axis=1)

# 分析集成结果的分布
print("Result distribution:")
print(merged['Fertilizer Name'].value_counts())

# 保存结果
submission = merged[['id', 'Fertilizer Name']]
submission.to_csv('submission.csv', index=False)
print("Submission file saved to submission.csv")

