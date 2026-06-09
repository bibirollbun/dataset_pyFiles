# 学号：2024423310216 姓名：王华锐
# ————————————————————————————
#
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
import joblib


def map_at_5(y_true, y_pred):
    """
    计算MAP@5的自定义评估函数
    """
    U = len(y_true)
    ap_sum = 0

    for i in range(U):
        rel_count = 0
        precision_sum = 0

        for k in range(5):
            if y_pred[i, k] == y_true[i]:
                rel_count += 1
                precision_at_k = rel_count / (k + 1)
                precision_sum += precision_at_k

        ap_sum += precision_sum / min(5, rel_count) if rel_count > 0 else 0

    return ap_sum / U


def main():

    # 数据加载与预处理
    print("步骤1: 数据加载与预处理...")

    # 加载数据集
    train_data = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
    test_data = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

    # 特征工程
    # 示例：创建养分比例特征
    train_data['N_P_ratio'] = train_data['Nitrogen'] / (train_data['Phosphorous'] + 1e-6)
    test_data['N_P_ratio'] = test_data['Nitrogen'] / (test_data['Phosphorous'] + 1e-6)

    # 特征列表
    features = ['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type',
                'Nitrogen', 'Potassium', 'Phosphorous', 'N_P_ratio']

    # 标签编码分类特征
    categorical_cols = ['Soil Type', 'Crop Type']
    for col in categorical_cols:
        le = LabelEncoder()
        train_data[col] = le.fit_transform(train_data[col])
        test_data[col] = le.transform(test_data[col])
        joblib.dump(le, f'{col}_encoder.pkl')  # 保存编码器

    # 分离特征和目标
    X = train_data[features]
    y = train_data['Fertilizer Name']

    # 编码目标变量
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    joblib.dump(label_encoder, 'label_encoder.pkl')

    # 模型训练

    print("步骤2: 模型训练...")

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    # 初始化模型
    model = LGBMClassifier(
        objective='multiclass',
        num_class=len(label_encoder.classes_),
        n_estimators=1000,
        learning_rate=0.05,
        random_state=42,
        metric='multi_logloss',
        verbose=50,  # 每50次迭代显示一次日志
        early_stopping_rounds=50  # 将early_stopping_rounds放在这里
    )

    # 训练模型
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)]
    )


    # 模型评估
    print("步骤3: 模型评估...")

    # 验证集预测
    val_probs = model.predict_proba(X_val)
    top5_preds = np.argsort(-val_probs, axis=1)[:, :5]

    # 计算MAP@5
    map5_score = map_at_5(y_val, top5_preds)
    print(f"验证集 MAP@5: {map5_score:.4f}")

    # 测试集预测

    print("步骤4: 测试集预测...")

    # 测试集预测
    X_test = test_data[features]
    test_probs = model.predict_proba(X_test)
    top5_indices = np.argsort(-test_probs, axis=1)[:, :5]

    # 转换回原始标签
    top5_labels = label_encoder.inverse_transform(top5_indices.flatten())
    top5_labels = top5_labels.reshape(len(X_test), 5)

    # 创建提交格式
    submission = pd.DataFrame({
        'id': test_data['id'],
        'Fertilizer Name': [' '.join(row) for row in top5_labels]
    })

    # 保存结果
    submission.to_csv('submission.csv', index=False)
    joblib.dump(model, 'fertilizer_model.pkl')
    print("预测结果已保存至 submission.csv")


    # 特征重要性分析
    print("步骤5: 特征重要性分析...")
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\n特征重要性排名:")
    print(feature_importance)



if __name__ == "__main__":
    main()

