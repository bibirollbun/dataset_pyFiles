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


train_df = pd.read_csv('/kaggle/input/credit-card-fraud-prediction/train.csv')
test_df = pd.read_csv('/kaggle/input/credit-card-fraud-prediction/test.csv')
sample_df = pd.read_csv('/kaggle/input/credit-card-fraud-prediction/sample_submission.csv')


train_df['IsFraud'].value_counts()


test_df['Time'] = pd.to_datetime(test_df['Time'], unit='s')  # 将时间戳转为 datetime 格式
# 提取时间特征
test_df['hour'] = test_df['Time'].dt.hour
test_df['minute'] = test_df['Time'].dt.minute  # 这里修正为提取分钟


train_df['Time'] = pd.to_datetime(train_df['Time'], unit='s')  # 将时间戳转为 datetime 格式
# 提取时间特征
train_df['hour'] = train_df['Time'].dt.hour
train_df['minute'] = train_df['Time'].dt.minute  # 这里修正为提取分钟


train_feature = train_df.drop(columns=['id','IsFraud','Time'])
test_feature = test_df.drop(columns=['id','Time'])

label = train_df['IsFraud']


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 计算相关性矩阵
correlation_matrix = train_feature.corr()

# 设置热图的大小
plt.figure(figsize=(15, 15))

# 绘制热图
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".1f", linewidths=0.5)

# 设置标题
plt.title('Correlation Heatmap of test_feature')

# 显示图形
plt.show()



from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import optuna

x = train_feature
y = label

# 切分数据集
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 定义目标函数
def objective(trial):
    # 计算正负样本的比例
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    # 参数空间
    params = {
        'objective': 'binary',  # 二分类
        'scale_pos_weight': scale_pos_weight,  # 优化scale_pos_weight
        'boosting_type': 'gbdt',  # 使用传统的 GBDT
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10.0),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.1),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0),
    }
    
    model = lgb.LGBMClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    
    # 获取预测的类别1的概率
    y_proba = model.predict_proba(X_test)[:, 1]  # 取类别 1 的概率

    # 计算 AUC
    auc = roc_auc_score(y_test, y_proba)
    return auc

# 启动优化
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# 使用最佳参数训练模型
best_params = study.best_trial.params
best_model = lgb.LGBMClassifier(**best_params, random_state=42)
best_model.fit(X_train, y_train)

# 评估
y_proba = best_model.predict_proba(X_test)[:, 1]  # 获取类别1的概率
auc = roc_auc_score(y_test, y_proba)

print("AUC分数: {:.5f}".format(auc))



# 使用模型进行预测
predictions = best_model.predict_proba(test_feature)[:, 1]

ids = test_df['id'].copy()

# 创建一个 DataFrame，将预测结果和 id 组合在一起
result = pd.DataFrame({
    'id': ids,
    'IsFraud': predictions
})

result.to_csv('prediction_results.csv', index=False)
result

