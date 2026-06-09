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


# === Step 1: 数据加载 ===
# 修改路径为 Kaggle 提供的绝对路径     without gpu   
train_path = '/kaggle/input/ventilator-pressure-prediction/train.csv'
test_path = '/kaggle/input/ventilator-pressure-prediction/test.csv'

dtypes = {
    'id': 'int32',
    'breath_id': 'int32',
    'R': 'int8',        # 只有 5, 20, 50
    'C': 'int8',        # 只有 10, 20, 50
    'time_step': 'float32',
    'u_in': 'float32',
    'u_out': 'int8',    # 0 或 1
    'pressure': 'float32'
}

# 使用修改后的路径读取数据
train = pd.read_csv(train_path, dtype=dtypes)
test = pd.read_csv(test_path, dtype=dtypes)


train


test


# 合并数据以便统一处理
data = pd.concat([train, test], axis=0, ignore_index=True)
print("数据加载完成，总样本数:", len(data))


# === Step 2: 特征工程 ===
def add_features(df):
    df['time_step'] = df['time_step'].astype(float)
    df['breath_id_lag'] = df['breath_id'].shift(1)
    df['breath_id_lag_diff'] = (df['breath_id'] != df['breath_id'].shift(1)).astype(int)
    df['u_in_cumsum'] = df.groupby('breath_id')['u_in'].cumsum()

    # Lag 特征
    df['u_in_lag1'] = df.groupby('breath_id')['u_in'].shift(1)
    df['u_in_lag2'] = df.groupby('breath_id')['u_in'].shift(2)
    df['u_in_lag3'] = df.groupby('breath_id')['u_in'].shift(3)
    
    # Rolling 特征
    df['u_in_rolling_mean_3'] = df.groupby('breath_id')['u_in'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['u_in_rolling_std_3'] = df.groupby('breath_id')['u_in'].transform(lambda x: x.rolling(3, min_periods=1).std())
    
    # Integral 特征（近似积分）
    df['u_in_integral'] = df['u_in'] * 0.1  # 0.1秒间隔
    df['u_in_integral'] = df.groupby('breath_id')['u_in_integral'].cumsum()
    
    # 呼吸周期内时间特征
    df['time_from_start'] = df.groupby('breath_id')['time_step'].transform(lambda x: x - x.min())
    
    return df

data = add_features(data)


# 分割训练集和测试集
train = data[data['pressure'].notnull()].reset_index(drop=True)
test = data[data['pressure'].isnull()].reset_index(drop=True)

# 构造特征和标签
features = [
    'R', 'C', 'time_step', 'u_in', 'u_out',
    'u_in_lag1', 'u_in_lag2', 'u_in_lag3',
    'u_in_rolling_mean_3', 'u_in_rolling_std_3',
    'u_in_cumsum', 'u_in_integral', 'time_from_start'
]
X_train = train[features]
y_train = train['pressure']
X_test = test[features]


#  I forget import package
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# === Step 3: 交叉验证与模型训练 ===
params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'verbose': -1
}

# 使用 GroupKFold 避免同一 breath_id 拆分到不同 fold
gkf = GroupKFold(n_splits=5)
oof_preds = np.zeros(len(X_train))
models = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=train['breath_id'])):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    print(f"\nFold {fold + 1} 开始训练...")
    model = lgb.train(params, train_data, num_boost_round=500,
                      valid_sets=[val_data], 
                     callbacks=[
        lgb.early_stopping(stopping_rounds=30, verbose=True),  # 替代 early_stopping_rounds
        lgb.log_evaluation(period=100)                         # 替代 verbose_eval
    ]
                     )
    models.append(model)
    
    oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    
    # 打印验证 MAE
    val_score = mean_absolute_error(y_val, oof_preds[val_idx])
    print(f"Fold {fold + 1} MAE: {val_score:.5f}")


# === Step 4: 预测与提交 ===
test_ids = test['id'].values
preds = np.zeros((len(models), len(X_test)))

for i, model in enumerate(models):
    preds[i] = model.predict(X_test, num_iteration=model.best_iteration)

final_preds = preds.mean(axis=0)

submission = pd.DataFrame({
    'id': test_ids,
    'pressure': final_preds
})


# 保存提交文件
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("提交文件已生成：submission.csv")

# 可视化特征重要性（可选）
if len(models) > 0:
    feature_importance = models[0].feature_importance(importance_type='gain')
    feature_names = models[0].feature_name()
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, feature_importance)
    plt.xlabel("Feature Importance (Gain)")
    plt.title("LightGBM Feature Importance")
    plt.show()


# 获取特征重要性（以 gain 为标准）
feature_importance = models[0].feature_importance(importance_type='gain')
feature_names = models[0].feature_name()

# 组合成 (特征名, 重要性) 的列表，并按重要性排序（从高到低）
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values(by='importance', ascending=False)

# 显示结果
print(importance_df)




