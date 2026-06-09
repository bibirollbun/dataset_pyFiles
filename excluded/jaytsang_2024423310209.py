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


# ========== 导入库 ==========
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import classification_report
import warnings

# ========== 设置 ==========
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ========== 数据加载 ==========
train_df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

# 显示数据基本信息
print("训练集形状:", train_df.shape)
print("测试集形状:", test_df.shape)
print("\n训练集前5行:")
display(train_df.head())
print("\n测试集前5行:")
display(test_df.head())                                                                                                                                         # 数据探索分析
def explore_data(df, name):
    print(f"\n===== {name}数据集分析 =====")
    print("\n1. 基本信息:")
    print(df.info())
    
    print("\n2. 描述统计:")
    display(df.describe())
    
    print("\n3. 缺失值检查:")
    print(df.isnull().sum())
    
    if 'fertilizer' in df.columns:
        print("\n4. 肥料类型分布:")
        display(df['fertilizer'].value_counts())

explore_data(train_df, "训练集")
explore_data(test_df, "测试集")        
# ========== 数据预处理 ==========
def preprocess_data(train_df, test_df):
    # 统一列名格式
    train_df.columns = [col.replace(' ', '_') for col in train_df.columns]
    test_df.columns = [col.replace(' ', '_') for col in test_df.columns]
    
    # 定义关键列
    id_col = 'id'
    target_col = 'Fertilizer_Name'
    
    # 特征工程
    def create_features(df):
        for col in ['Soil_Type', 'Crop_Type']:
            if col in df.columns:
                df[col] = df[col].astype('category').cat.codes
        
        df['N_P_ratio'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-6)
        df['N_K_ratio'] = df['Nitrogen'] / (df['Potassium'] + 1e-6)
        df['P_K_ratio'] = df['Phosphorous'] / (df['Potassium'] + 1e-6)
        df['NPK_total'] = df['Nitrogen'] + df['Phosphorous'] + df['Potassium']
        df['temp_humidity_interaction'] = df['Temparature'] * df['Humidity']
        df['soil_moisture_interaction'] = df['Soil_Type'] * df['Moisture']
        return df
    
    train_df = create_features(train_df)
    test_df = create_features(test_df)
    
    # 编码目标变量
    le = LabelEncoder()
    train_df[target_col] = le.fit_transform(train_df[target_col])
    
    # 准备数据
    X = train_df.drop([id_col, target_col], axis=1)
    y = train_df[target_col]
    X_test = test_df.drop([id_col], axis=1)
    X_test = X_test[X.columns]
    
    return X, y, X_test, le, test_df[id_col]

X, y, X_test, label_encoder, test_ids = preprocess_data(train_df, test_df)

# ========== 训练模型 ==========
def train_model(X, y, X_test):
    params = {
        'objective': 'multiclass',
        'num_class': len(np.unique(y)),
        'metric': 'multi_logloss',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'random_state': 42,
        'verbose': -1
    }
    
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    test_preds = np.zeros((X_test.shape[0], len(np.unique(y))))
    models = []
    
    for fold, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
        print(f"Training fold {fold+1}")
        X_trn, y_trn = X.iloc[trn_idx], y.iloc[trn_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        train_data = lgb.Dataset(X_trn, label=y_trn)
        valid_data = lgb.Dataset(X_val, label=y_val)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, valid_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(100)
            ]
        )
        test_preds += model.predict(X_test) / folds.n_splits
        models.append(model)
    
    return test_preds, models

test_preds, models = train_model(X, y, X_test)

# ========== 模型评估 ==========
def evaluate_model(model, X_val, y_val, label_encoder):
    y_pred_proba = model.predict(X_val)
    top5_indices = np.argsort(y_pred_proba, axis=1)[:, -5:][:, ::-1]
    
    def map_at_5(y_true, y_pred_top5):
        map_score = 0.0
        for u in range(len(y_true)):
            ap = 0.0
            num_correct = 0
            for k in range(5):
                if y_pred_top5[u, k] == y_true.iloc[u]:
                    num_correct += 1
                    ap += num_correct / (k + 1)
            if num_correct > 0:
                ap /= num_correct
            map_score += ap
        return map_score / len(y_true)
    
    map_score = map_at_5(y_val, top5_indices)
    print(f"MAP@5分数: {map_score:.4f}")
    
    plt.figure(figsize=(12, 8))
    lgb.plot_importance(model, max_num_features=20)
    plt.title("特征重要性")
    plt.show()
    
    return map_score

# 使用第一个模型进行评估
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
evaluate_model(models[0], X_val, y_val, label_encoder)

# ========== 生成提交文件 ==========
def create_submission(preds, ids, encoder):
    top5_indices = np.argsort(preds, axis=1)[:, -5:][:, ::-1]
    top5_labels = encoder.inverse_transform(top5_indices.reshape(-1)).reshape(top5_indices.shape)
    
    submission = pd.DataFrame({'id': ids})
    for i in range(5):
        submission[f'Fertilizer_Name_{i+1}'] = top5_labels[:, i]
    return submission

submission = create_submission(test_preds, test_ids, label_encoder)
submission.to_csv('submission.csv', index=False)
print("提交文件已生成:")
display(submission.head())


display(submission.head())

