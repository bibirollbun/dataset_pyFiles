print("# 学号: 2024423310204, 姓名: 陈奕希")
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 加载数据
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

# 数据预处理
X = train.drop('Fertilizer Name', axis=1)
y = train['Fertilizer Name']
X_test = test.copy()

# 处理数值特征
num_cols = X.select_dtypes(include=['float64', 'int64']).columns
X[num_cols] = StandardScaler().fit_transform(X[num_cols])
X_test[num_cols] = StandardScaler().fit_transform(X_test[num_cols])

# 处理类别特征
cat_cols = X.select_dtypes(include=['object']).columns
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    X_test[col] = le.transform(X_test[col])

# 编码目标变量
y_encoder = LabelEncoder()
y_encoded = y_encoder.fit_transform(y)
classes = y_encoder.classes_

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 训练XGBoost模型
model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=len(np.unique(y_encoded)),
    eval_metric=['mlogloss'],
    n_estimators=300,
    learning_rate=0.1,
    max_depth=4,
    random_state=42,
    early_stopping_rounds=30,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50
)

# 生成预测结果（改为直接预测类别，而非Top5）
y_test_pred = model.predict(X_test)
y_test_pred_labels = y_encoder.inverse_transform(y_test_pred)  # 转回原始类别名称

# 创建符合竞赛要求的提交文件
submission = test[['id']].copy()  # 确保测试集的ID列名正确
submission['Fertilizer Name'] = y_test_pred_labels  # 单列预测结果

submission.to_csv('submission.csv', index=False)
print("预测结果已保存至submission.csv")

