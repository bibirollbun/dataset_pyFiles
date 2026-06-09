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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

# 加载数据
train_data = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

# 检查数据
print(train_data.head())
print(train_data.info())
print(train_data['Fertilizer Name'].value_counts())  # 查看标签分布


from sklearn.preprocessing import StandardScaler

numeric_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
scaler = StandardScaler()
train_data[numeric_features] = scaler.fit_transform(train_data[numeric_features])

# 类别特征编码（土壤类型）
soil_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
soil_encoded = soil_encoder.fit_transform(train_data[['Soil Type']])
soil_encoded_df = pd.DataFrame(soil_encoded, columns=soil_encoder.get_feature_names_out(['Soil Type']))

crop_encoder = LabelEncoder()
if 'Crop Type' in train_data.columns:
    train_data['Crop Type Encoded'] = crop_encoder.fit_transform(train_data['Crop Type'])
    crop_categories = crop_encoder.classes_  # 保存类别顺序用于后续解码

# 合并特征
X = pd.concat([
    train_data[numeric_features],
    soil_encoded_df
], axis=1)

# 标签编码
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(train_data['Fertilizer Name'])
y_categories = label_encoder.classes_ 


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


import lightgbm as lgb
from sklearn.metrics import accuracy_score

# 创建LightGBM数据集
train_data_lgb = lgb.Dataset(X_train, label=y_train)
val_data_lgb = lgb.Dataset(X_val, label=y_val, reference=train_data_lgb)

# 模型参数
params = {
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class': len(y_categories),
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'random_state': 42
}

# 训练模型
lgb_model = lgb.train(
    params,
    train_data_lgb,
    valid_sets=[train_data_lgb, val_data_lgb],
    num_boost_round=500,
    callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=50)
        ]
)

# 验证集预测
val_probs = lgb_model.predict(X_val)
val_pred = np.argmax(val_probs, axis=1)
print(f"Validation Accuracy: {accuracy_score(y_val, val_pred):.4f}")


def get_topk_predictions(probs, k=5):
    """获取概率最高的k个类别"""
    topk_indices = np.argsort(probs, axis=1)[:, -k:][:, ::-1]
    return [[y_categories[i] for i in indices] for indices in topk_indices]

# 示例：获取验证集前5预测
val_top5 = get_topk_predictions(val_probs, k=5)


import matplotlib.pyplot as plt

# 获取特征重要性
feature_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importance': lgb_model.feature_importance()
}).sort_values('Importance', ascending=False)

# 可视化
plt.figure(figsize=(10, 6))
plt.barh(feature_imp['Feature'], feature_imp['Importance'])
plt.xlabel('Feature Importance')
plt.title('LightGBM Feature Importance')
plt.gca().invert_yaxis()
plt.show()

# 关键发现示例
print("\nTop 5 Important Features:")
print(feature_imp.head(5))


from xgboost import XGBClassifier

# XGBoost实现
xgb_model = XGBClassifier(
    objective='multi:softprob',
    num_class=len(y_categories),
    learning_rate=0.05,
    n_estimators=500,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# 评估
xgb_probs = xgb_model.predict_proba(X_val)
xgb_pred = np.argmax(xgb_probs, axis=1)
print(f"XGBoost Validation Accuracy: {accuracy_score(y_val, xgb_pred):.4f}")


# 创建两个模型的预测平均
def ensemble_predict(model1_probs, model2_probs):
    """简单概率平均集成"""
    return (model1_probs + model2_probs) / 2

ensemble_probs = ensemble_predict(val_probs, xgb_probs)
ensemble_pred = np.argmax(ensemble_probs, axis=1)
print(f"Ensemble Validation Accuracy: {accuracy_score(y_val, ensemble_pred):.4f}")


# 测试集预处理
test_numeric = scaler.transform(test_data[numeric_features])
test_soil_encoded = soil_encoder.transform(test_data[['Soil Type']])
test_soil_df = pd.DataFrame(test_soil_encoded, columns=soil_encoder.get_feature_names_out(['Soil Type']))

X_test = pd.concat([
    pd.DataFrame(test_numeric, columns=numeric_features),
    test_soil_df
], axis=1)

test_probs_lgb = lgb_model.predict(X_test)  # LightGBM 预测测试集
test_probs_xgb = xgb_model.predict_proba(X_test)  # XGBoost 预测测试集
 
test_probs_ensemble = ensemble_predict(test_probs_lgb, test_probs_xgb)  # 集成预测
# 预测
test_top5 = get_topk_predictions(test_probs_ensemble, k=5)

# 保存结果
results = pd.DataFrame({
    'Id': test_data['id'],
    'Fertilizer Name':[' '.join(pred) for pred in test_top5]
})
results.to_csv('fertilizer_predictions.csv', index=False)


from sklearn.metrics import average_precision_score

def mapk(true_labels, pred_labels, k=5):
    """计算MAP@k"""
    aps = []
    for true, preds in zip(true_labels, pred_labels):
        # 将真实标签转换为one-hot编码位置
        true_pos = [i for i, label in enumerate(y_categories) if label == true][0]
        # 计算预测中的排名
        ranks = [i for i, label in enumerate(preds) if label in y_categories]
        ap = 1.0 if true_pos in ranks[:k] else 0.0
        aps.append(ap)
    return np.mean(aps)

val_true_labels = [y_categories[label] for label in y_val]
val_map5 = mapk(val_true_labels, val_top5)
print(f"Validation MAP@5: {val_map5:.4f}")

