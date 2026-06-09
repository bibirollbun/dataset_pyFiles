# 导入必要的库
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# 读取数据
train_df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
submission_df = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')

# 查看数据概览
print(train_df.head())
print(test_df.head())
print(submission_df.head())

# 特征工程：Label Encoding 对分类变量进行编码
label_encoder = LabelEncoder()

# 编码 'Soil Type' 和 'Crop Type'
train_df['Soil Type'] = label_encoder.fit_transform(train_df['Soil Type'])
test_df['Soil Type'] = label_encoder.transform(test_df['Soil Type'])

train_df['Crop Type'] = label_encoder.fit_transform(train_df['Crop Type'])
test_df['Crop Type'] = label_encoder.transform(test_df['Crop Type'])

# 编码 'Fertilizer Name'（目标列）
fertilizer_encoder = LabelEncoder()
train_df['Fertilizer Name'] = fertilizer_encoder.fit_transform(train_df['Fertilizer Name'])

# 目标列为 'Fertilizer Name'（目标列名称）
X = train_df.drop(['id', 'Fertilizer Name'], axis=1)  # 去掉 'id' 和 'Fertilizer Name' 作为特征
y = train_df['Fertilizer Name']  # 'Fertilizer Name' 作为目标

# 随机选择训练集的一部分进行加速（仅用于快速调试）
X_train_small, _, y_train_small, _ = train_test_split(X, y, test_size=0.9, random_state=42)

# 切分数据集（70% 训练，30% 测试）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 模型定义（先使用 RandomForest 进行快速测试）
models = {
    'RandomForest': RandomForestClassifier(n_estimators=10, random_state=42),
    'LightGBM': lgb.LGBMClassifier(n_estimators=50),
    'XGBoost': xgb.XGBClassifier(objective='multi:softmax', eval_metric='mlogloss', n_estimators=50, tree_method='hist'),  # CPU-based training
    'CatBoost': cb.CatBoostClassifier(iterations=50, learning_rate=0.1, depth=6, verbose=0, task_type='CPU')  # Use CPU for CatBoost
}

# 定义模型评估函数
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)  # 获取预测类别
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# 评估模型
results = {}
for model_name, model in models.items():
    accuracy = evaluate_model(model, X_train_small, X_test, y_train_small, y_test)
    results[model_name] = accuracy

# 输出评估结果
print("Model Evaluation Results:")
for model_name, score in results.items():
    print(f"{model_name}: Accuracy = {score:.4f}")

# 超参数调优（以 LightGBM 为例）
param_grid = {
    'num_leaves': [31, 50],
    'learning_rate': [0.1],
    'n_estimators': [50]
}

grid_search = GridSearchCV(lgb.LGBMClassifier(), param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print("Best Parameters for LightGBM:", grid_search.best_params_)

# 模型预测（用最佳模型做预测）
best_model = grid_search.best_estimator_
y_test_pred = best_model.predict(X_test)

# 在测试集上生成预测
test_predictions = best_model.predict(test_df.drop(['id'], axis=1))

# 将预测结果转换回原始的 'Fertilizer Name' 标签
test_predictions = fertilizer_encoder.inverse_transform(test_predictions)

# 生成提交文件
submission_df['Fertilizer Name'] = test_predictions
submission_df.to_csv('submission.csv', index=False)

print("Submission file has been saved.")


