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


# 学号：2024423310109，姓名：陈妍妍
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import label_ranking_average_precision_score

def map5_score(y_true, y_pred):
    """计算MAP@5评估指标"""
    y_true_binary = np.zeros_like(y_pred)
    for i, label in enumerate(y_true):
        y_true_binary[i, label] = 1
    return label_ranking_average_precision_score(y_true_binary, y_pred)

def load_data():
    """加载并预处理数据"""
    dtype_spec = {
        'id': 'int32',
        'Nitrogen': 'float32',
        'Phosphorous': 'float32',
        'Potassium': 'float32',
        'Temperature': 'float32',
        'Humidity': 'float32',
        'Soil Type': 'category',
        'Crop Type': 'category',
        'Fertilizer Name': 'category'
    }
    
    # 读取数据
    train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv', dtype=dtype_spec)
    test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv', dtype=dtype_spec)
    
    # 特征工程：添加比值和总和特征
    def add_features(df):
        arr = df[['Nitrogen', 'Phosphorous', 'Potassium']].values
        df['NP_ratio'] = np.divide(arr[:, 0], arr[:, 1], out=np.zeros(arr.shape[0]), where=arr[:, 1] != 0)
        df['NK_ratio'] = np.divide(arr[:, 0], arr[:, 2], out=np.zeros(arr.shape[0]), where=arr[:, 2] != 0)
        df['PK_ratio'] = np.divide(arr[:, 1], arr[:, 2], out=np.zeros(arr.shape[0]), where=arr[:, 2] != 0)
        df['Nutrient_sum'] = arr.sum(axis=1)
        return df
    
    train = add_features(train)
    test = add_features(test)
    
    # 目标编码
    target = 'Fertilizer Name'
    le = LabelEncoder()
    y = le.fit_transform(train[target])
    
    # 分类特征编码
    cat_cols = ['Soil Type', 'Crop Type']
    for col in cat_cols:
        train[col] = train[col].astype('category').cat.codes
        test[col] = test[col].astype('category').cat.codes
    
    X = train.drop([target, 'id'], axis=1)
    X_test = test.drop(['id'], axis=1)
    
    return X, y, X_test, le, cat_cols

def train_model(X, y, X_test, le, cat_cols):
    """训练多分类梯度提升树模型"""
    params = {
        'objective': 'multiclass',
        'num_class': len(le.classes_),
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'min_child_samples': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'verbosity': -1,
        'seed': 42
    }
    
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    test_preds = np.zeros((X_test.shape[0], len(le.classes_)))
    oof_preds = np.zeros((X.shape[0], len(le.classes_)))
    
    for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
        print(f"\nFold {fold + 1}")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
        val_set = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols)
        
        model = lgb.train(
            params,
            train_set,
            num_boost_round=1000,
            valid_sets=[train_set, val_set],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(100)
            ]
        )
        
        # 验证集预测和评估
        val_pred = model.predict(X_val)
        score = map5_score(y_val, val_pred)
        print(f"Fold {fold + 1} MAP@5: {score:.5f}")
        
        # 累积测试集预测
        test_preds += model.predict(X_test) / folds.n_splits
        oof_preds[val_idx] = val_pred
    
    # 整体OOF评估
    oof_score = map5_score(y, oof_preds)
    print(f"\nOverall OOF MAP@5: {oof_score:.5f}")
    
    return test_preds, model  # 返回模型用于特征重要性分析

def create_submission(preds, le, test_df):
    """生成Top5预测结果的提交文件"""
    top5_indices = np.argsort(-preds, axis=1)[:, :5]  # 获取Top5索引
    top5_labels = le.inverse_transform(top5_indices.flatten()).reshape(top5_indices.shape)
    
    submission = pd.DataFrame({
        'id': test_df['id'],
        'Top5 Predictions': [', '.join(row) for row in top5_labels]  # 逗号分隔
    })
    
    print("\nSubmission Sample:")
    print(submission.head())
    return submission

def feature_importance_analysis(model, X):
    """特征重要性分析"""
    importance = model.feature_importance(importance_type='gain')
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(importance_df)
    return importance_df

if __name__ == "__main__":
    print("Loading and preprocessing data...")
    X, y, X_test, le, cat_cols = load_data()
    
    print("\nTraining model...")
    test_preds, model = train_model(X, y, X_test, le, cat_cols)
    
    print("\nGenerating submission...")
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
    submission = create_submission(test_preds, le, test_df)
    submission.to_csv('submission.csv', index=False)
    
    print("\nFeature importance analysis...")
    importance_df = feature_importance_analysis(model, X)
    importance_df.to_csv('feature_importance.csv', index=False)
    
    print("\nDone! Submission and feature importance saved.")




