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
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')  
    test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')   
    sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')  
    target_column = 'Fertilizer Name'  
    X = train.drop(target_column, axis=1)
    y = train[target_column]
    return X, y, test, sample_submission

def preprocess_data(X, y, X_test):
    cat_features = X.select_dtypes(include=['object']).columns
    num_features = X.select_dtypes(include=['float64', 'int64']).columns

    # 类别特征编码
    for feature in cat_features:
        le = LabelEncoder()
        X[feature] = le.fit_transform(X[feature])
        if feature in X_test.columns:
            X_test[feature] = le.transform(X_test[feature])

    # 数值特征标准化
    scaler = StandardScaler()
    X[num_features] = scaler.fit_transform(X[num_features])
    X_test[num_features] = scaler.transform(X_test[num_features])

    # 保存 LabelEncoder 实例（关键修复）
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return X, y_encoded, X_test, le  # 返回 le 而不是 classes

def build_and_train_model(X_train, y_train, X_val, y_val):
    model = xgb.XGBClassifier(
        objective='multi:softprob',  
        num_class=len(np.unique(y_train)),  
        eval_metric=['mlogloss'],  
        n_estimators=500,  
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.01,
        reg_lambda=0.01,
        use_label_encoder=False,
        verbosity=0,  
        early_stopping_rounds=50  
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],  
        verbose=200  
    )
    print(f"最优迭代轮次: {model.best_iteration}, 验证集最小损失: {model.best_score}")
    return model

def calculate_map5(y_true_onehot, y_pred_proba, classes):
    map5_score = 0.0
    for i in range(len(y_true_onehot)):
        relevant = 0
        precision_sum = 0
        for k in range(5):
            if k < len(classes):
                pred_label = classes[np.argsort(y_pred_proba[i])[::-1][k]]
                pred_index = np.where(classes == pred_label)[0][0]
                if y_true_onehot[i, pred_index] == 1 and relevant == 0:
                    relevant = 1
                    precision = relevant / (k + 1)
                    precision_sum += precision
        map5_score += precision_sum / min(5, len(classes))
    return map5_score / len(y_true_onehot)

def evaluate_model(model, X_val, y_val, le):  # 接收 le 实例
    y_pred_proba = model.predict_proba(X_val)
    
    # 解码为原始标签（关键修复）
    y_val_original = le.inverse_transform(y_val)  
    classes = le.classes_
    
    y_true_onehot = np.zeros((len(y_val_original), len(classes)))
    for i, label in enumerate(y_val_original):
        # 安全获取索引（双重校验）
        label_indices = np.where(classes == label)[0]
        if len(label_indices) == 0:
            raise ValueError(f"标签 {label} 不在类别列表中！")
        y_true_onehot[i, label_indices[0]] = 1  # 使用 [0] 避免多层索引
    
    map5 = calculate_map5(y_true_onehot, y_pred_proba, classes)
    print(f"MAP@5 Score: {map5:.4f}")
    return map5

def predict_top5(model, X_test, le):  # 接收 le 实例
    y_pred_proba = model.predict_proba(X_test)
    classes = le.classes_
    top5_indices = np.argsort(y_pred_proba, axis=1)[:, -5:][:, ::-1]
    return [[classes[i] for i in row] for row in top5_indices]

def feature_importance_analysis(model, X, feature_names):
    importance = model.feature_importances_
    df_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=df_importance.head(10))
    plt.title('Top 10 Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    return df_importance

def main():
    X, y, X_test, sample_submission = load_data()
    X, y_encoded, X_test, le = preprocess_data(X, y, X_test)  # 接收 le 实例
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    
    model = build_and_train_model(X_train, y_train, X_val, y_val)
    evaluate_model(model, X_val, y_val, le)  # 传入 le 实例
    feature_importance_analysis(model, X_train, X_train.columns)
    
    top5_predictions = predict_top5(model, X_test, le)  # 传入 le 实例
    
    submission = pd.DataFrame({
        'id': sample_submission['id'],  
        'Fertilizer Name': [','.join(pred) for pred in top5_predictions]
    })
    submission.to_csv('/kaggle/working/submission.csv', index=False)
    print("提交文件已生成")

if __name__ == "__main__":
    main()

