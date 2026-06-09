#学号：2024423310104     姓名：陈明聪
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


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import top_k_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def load_data():
    train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')  
    test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')   
    sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')  
    return train, test, sample_submission

def preprocess_data(train, test):
    # 分离特征和目标
    target = 'Fertilizer Name'
    X = train.drop(columns=[target])
    y = train[target]
    X_test = test.copy()
    
    # 统一处理类别特征
    cat_features = X.select_dtypes(include=['object']).columns
    for col in cat_features:
        # 合并训练测试集确保编码一致性
        combined = pd.concat([X[col], X_test[col]], axis=0)
        le = LabelEncoder().fit(combined)
        X[col] = le.transform(X[col])
        X_test[col] = le.transform(X_test[col])
    
    # 数值特征标准化
    num_features = X.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    X[num_features] = scaler.fit_transform(X[num_features])
    X_test[num_features] = scaler.transform(X_test[num_features])
    
    # 目标编码
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    
    return X, y_encoded, X_test, le_target

def build_and_train_model(X_train, y_train, X_val, y_val, num_classes):
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=num_classes,
        eval_metric='mlogloss',
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        use_label_encoder=False,
        early_stopping_rounds=50,
        random_state=42
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    
    print(f"Best iteration: {model.best_iteration}, Val loss: {model.best_score:.4f}")
    return model

def evaluate_model(model, X_val, y_val, le_target):
    # 预测概率
    y_pred_proba = model.predict_proba(X_val)
    
    # 使用top-k准确率评估
    top1_acc = top_k_accuracy_score(y_val, y_pred_proba, k=1)
    top5_acc = top_k_accuracy_score(y_val, y_pred_proba, k=5)
    print(f"Top-1 Accuracy: {top1_acc:.4f}")
    print(f"Top-5 Accuracy: {top5_acc:.4f}")
    
    # 计算MAP@5
    map5 = calculate_map5(y_val, y_pred_proba)
    print(f"MAP@5 Score: {map5:.4f}")
    return map5

def calculate_map5(y_true, y_pred_proba):
    """优化后的MAP@5计算函数"""
    map5_scores = []
    for true_idx, probs in zip(y_true, y_pred_proba):
        # 获取排序后的索引 (从高到低)
        sorted_idx = np.argsort(probs)[::-1]
        # 检查真实标签在top5中的位置
        rank = np.where(sorted_idx == true_idx)[0]
        if rank.size > 0 and rank[0] < 5:
            # 位置从1开始计数 (rank=0 -> 1/1, rank=1 -> 1/2)
            map5_scores.append(1.0 / (rank[0] + 1))
        else:
            map5_scores.append(0.0)
    return np.mean(map5_scores)

def predict_top5(model, X_test, le_target):
    y_pred_proba = model.predict_proba(X_test)
    top5_indices = np.argsort(y_pred_proba, axis=1)[:, -5:][:, ::-1]
    return [list(le_target.inverse_transform(indices)) for indices in top5_indices]

def feature_importance_analysis(model, feature_names):
    importance = model.feature_importances_
    df_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=df_importance.head(15))
    plt.title('Top 15 Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    return df_importance

def main():
    # 加载数据
    train, test, sample_submission = load_data()
    
    # 预处理
    X, y, X_test, le_target = preprocess_data(train, test)
    
    # 划分数据集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 训练模型 (修复括号问题)
    model = build_and_train_model(
        X_train, y_train, 
        X_val, y_val,
        num_classes=len(le_target.classes_)
    )  # 添加了这行的右括号
    
    # 评估模型
    evaluate_model(model, X_val, y_val, le_target)
    
    # 特征重要性分析
    feature_importance_analysis(model, X.columns)
    
    # 生成预测
    top5_predictions = predict_top5(model, X_test, le_target)
    
    # 创建提交文件
    submission = sample_submission.copy()
    submission['Fertilizer Name'] = [','.join(pred) for pred in top5_predictions]
    submission.to_csv('submission.csv', index=False)
    print("Submission file created")
    
    # 保存模型和编码器
    joblib.dump(model, 'xgb_model.pkl')
    joblib.dump(le_target, 'label_encoder.pkl')

if __name__ == "__main__":
    main()

