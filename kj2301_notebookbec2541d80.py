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


# 快速优化版本 - 基于上次的快速代码
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

# 加载数据
train_data = pd.read_csv('/kaggle/input/predicting-equipment-failure/train.csv')
test_data = pd.read_csv('/kaggle/input/predicting-equipment-failure/test.csv')

print("数据形状:", train_data.shape, test_data.shape)

# 快速数据预处理
def quick_preprocess(train_df, test_df):
    train_processed = train_df.copy()
    test_processed = test_df.copy()
    
    # 分离特征和目标
    X = train_processed.drop('Health index', axis=1)
    y = train_processed['Health index']
    X_test = test_processed
    
    # 移除index列
    if 'index' in X.columns:
        X = X.drop('index', axis=1)
    if 'index' in X_test.columns:
        X_test = X_test.drop('index', axis=1)
    
    # 快速处理分类变量
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        le = LabelEncoder()
        combined = pd.concat([X[col], X_test[col]], axis=0)
        le.fit(combined)
        X[col] = le.transform(X[col])
        X_test[col] = le.transform(X_test[col])
    
    return X, y, X_test

X, y, X_test = quick_preprocess(train_data, test_data)

# 快速特征工程 - 只添加最重要的特征
def quick_features(df):
    df_engineered = df.copy()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 只添加最有效的统计特征
    if len(numerical_cols) > 0:
        df_engineered['mean_val'] = df[numerical_cols].mean(axis=1)
        df_engineered['std_val'] = df[numerical_cols].std(axis=1)
    
    # 只添加前2个数值特征的交互
    if len(numerical_cols) >= 2:
        col1, col2 = numerical_cols[0], numerical_cols[1]
        df_engineered[f'{col1}_mul_{col2}'] = df_engineered[col1] * df_engineered[col2]
    
    return df_engineered

print("快速特征工程...")
X_engineered = quick_features(X)
X_test_engineered = quick_features(X_test)

print(f"特征工程后形状: {X_engineered.shape}")

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_engineered)
X_test_scaled = scaler.transform(X_test_engineered)

# 分割训练验证集
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}")

# 只使用最快且效果好的模型
models = {
    'LightGBM': LGBMRegressor(n_estimators=200, random_state=42, n_jobs=-1),  # 最快
    'XGBoost': XGBRegressor(n_estimators=200, random_state=42, n_jobs=-1),   # 第二快
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)  # 适量树
}

# 快速训练和评估
print("\n快速模型训练...")
results = {}

for name, model in models.items():
    print(f"训练 {name}...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    results[name] = mse
    print(f"{name} MSE: {mse:.4f}")

# 选择最佳模型
best_model_name = min(results, key=results.get)
print(f"\n最佳模型: {best_model_name}")

# 使用全部数据训练最佳模型
print("使用全部数据训练最佳模型...")
best_model = models[best_model_name]
best_model.fit(X_scaled, y)

# 预测
test_predictions = best_model.predict(X_test_scaled)

# 简单的后处理
train_min, train_max = y.min(), y.max()
test_predictions = np.clip(test_predictions, train_min, train_max)

print(f"\n预测范围: [{test_predictions.min():.2f}, {test_predictions.max():.2f}]")

# 创建提交文件
submission = pd.DataFrame({
    'index': test_data['index'],
    'Health index': test_predictions
})

submission_file = '/kaggle/working/submission.csv'
submission.to_csv(submission_file, index=False)
print(f"\n提交文件已保存: {submission_file}")
print(submission.head())

