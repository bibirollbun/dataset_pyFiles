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
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time

STUDENT_ID = "2024423320225"
STUDENT_NAME = "卫斯杰"

def load_and_preprocess_data():
    """加载数据并预处理，适配实际列名"""
    print("开始数据预处理...")
    start_time = time.time()
    
    train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
    test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
     
    print("训练数据形状:", train.shape)
    print("测试数据形状:", test.shape)
    print("训练集列名:", train.columns.tolist())
    print("测试集列名:", test.columns.tolist())
    
    required_train_cols = ['id', 'Fertilizer Name']
    missing_cols = [col for col in required_train_cols if col not in train.columns]
    if missing_cols:
        raise ValueError(f"训练集缺少必要列: {missing_cols}")
    
    numeric_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = train.select_dtypes(include=['object']).columns.tolist()
    
    for col in numeric_cols:
        if col != 'id':  
            median_val = train[col].median()
            train[col] = train[col].fillna(median_val)
            if col in test.columns:
                test[col] = test[col].fillna(median_val)
    
    label_col = 'Fertilizer Name'
    for col in categorical_cols:
        if col != label_col:  
            if col in test.columns:
                mode_val = train[col].mode()[0]
                train[col] = train[col].fillna(mode_val)
                test[col] = test[col].fillna(mode_val)
    
    nutrient_cols = ['Nitrogen', 'Potassium', 'Phosphorous']
    available_nutrients = [col for col in nutrient_cols if col in train.columns and col in test.columns]
    
    if 'Temparature' in train.columns and 'Temparature' in test.columns and \
       'Humidity' in train.columns and 'Humidity' in test.columns:
        train['temp_humidity'] = train['Temparature'] * (train['Humidity'] / 100)
        test['temp_humidity'] = test['Temparature'] * (test['Humidity'] / 100)
    
    if 'Moisture' in train.columns and 'Moisture' in test.columns:
        train['moisture_index'] = train['Moisture'] * train['Humidity'] / 100
        test['moisture_index'] = test['Moisture'] * test['Humidity'] / 100
    
    for col in categorical_cols:
        if col != label_col and col in test.columns:
            le = LabelEncoder()
            train[col] = le.fit_transform(train[col])
            test[col] = le.transform(test[col])
    
    print(f"数据预处理完成，耗时: {time.time() - start_time:.2f}秒")
    return train, test, label_col

def calculate_map_at_k(y_true, y_pred, k=5):
    """高效计算MAP@k指标"""
    map_score = 0.0
    for i in range(len(y_true)):
        true_label = y_true[i]
        pred_labels = y_pred[i][:k]
        hits = np.where(pred_labels == true_label)[0]
        if len(hits) > 0:
            first_hit = hits[0]
            map_score += 1.0 / (first_hit + 1)
    return map_score / len(y_true)

def train_xgboost_model(X_train, X_val, y_train, y_val, num_classes):
    """训练XGBoost模型，修复参数警告"""
    print("开始训练XGBoost模型...")
    start_time = time.time()
    
    params = {
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': num_classes,
        'learning_rate': 0.1,
        'max_depth': 4,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'hist',
        'grow_policy': 'lossguide',
        'seed': 42,
        'n_jobs': -1
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=300,  
        evals=[(dval, 'val')],
        early_stopping_rounds=30,
        verbose_eval=50
    )
    
    y_pred_proba = model.predict(dval)
    y_pred_top5 = np.argsort(-y_pred_proba, axis=1)[:, :5]
    map_5 = calculate_map_at_k(y_val, y_pred_top5)
    print(f"模型训练完成，耗时: {time.time() - start_time:.2f}秒，MAP@5: {map_5:.4f}")
    
    return model

def generate_submission(model, test_df, le, label_col):
    """生成提交文件"""
    print("生成预测结果...")
    start_time = time.time()
    X_test = test_df.drop('id', axis=1)
    X_test = X_test[X_test.columns.intersection(model.feature_names)]
    dtest = xgb.DMatrix(X_test)
    y_pred_proba = model.predict(dtest)
    y_pred_top5 = np.argsort(-y_pred_proba, axis=1)[:, :5]
    pred_labels = [' '.join(le.inverse_transform(pred)) for pred in y_pred_top5]
    submission = pd.DataFrame({'id': test_df['id'], 'Fertilizer Name': pred_labels})
    submission.to_csv('submission.csv', index=False)
    print(f"提交文件已生成，包含 {len(submission)} 条记录，耗时: {time.time() - start_time:.2f}秒")
    return submission

def main():
    print(f"学号: {STUDENT_ID}, 姓名: {STUDENT_NAME}")
    print("肥料类型预测竞赛 - 最终稳定版")
    
    try:
        train, test, label_col = load_and_preprocess_data()
    except ValueError as e:
        print("\n数据加载错误:")
        print(e)
        return
    
    drop_cols = ['id', label_col]
    X = train.drop(drop_cols, axis=1)
    y = train[label_col]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    print(f"训练集大小: {X_train.shape}, 验证集大小: {X_val.shape}, 类别数: {num_classes}")
    
    model = train_xgboost_model(X_train, X_val, y_train, y_val, num_classes)
    
    submission = generate_submission(model, test, le, label_col)
    print("\n执行完成！请将Notebook设为Public并提交URL")
    print(f"提交文件第一行: {submission.iloc[0]['id']}, {submission.iloc[0]['Fertilizer Name']}")

if __name__ == "__main__":
    main()

