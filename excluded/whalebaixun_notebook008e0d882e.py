#学号: 2024423320228, 姓名: 曾嘉浚
# 导入必要的库
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# 1. 加载数据
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')

# 2. 数据预处理
# 编码标签
le_label = LabelEncoder()
train['Fertilizer Name'] = le_label.fit_transform(train['Fertilizer Name'])

# 提取特征和标签
X = train.drop(['id', 'Fertilizer Name'], axis=1)
y = train['Fertilizer Name']
X_test = test.drop(['id'], axis=1)

# 编码分类变量
for col in ['Soil Type', 'Crop Type']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    X_test[col] = le.transform(X_test[col])
    
# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 构建 XGBoost 模型（支持 map@5）
model = XGBClassifier(
    objective='multi:softprob',
    eval_metric=['mlogloss'],
    n_estimators=800,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.7,
    tree_method='hist'  # 更快的训练方式
)

# 训练模型
model.fit(X_train, y_train, 
          eval_set=[(X_train, y_train), (X_val, y_val)],
          early_stopping_rounds=50,
          verbose=100)

# 获取验证集预测概率
y_pred_proba = model.predict_proba(X_val)

def calculate_map_k(y_true, y_pred_proba, k=5):
    """
    计算 MAP@k
    :param y_true: 真实标签 (n_samples,)
    :param y_pred_proba: 预测概率 (n_samples, n_classes)
    :param k: 取 Top-k 预测
    :return: MAP@k 分数
    """
    top_k_predictions = np.argsort(y_pred_proba, axis=1)[:, -k:][:, ::-1]  # 概率从高到低排列
    map_score = 0.0

    for i in range(len(y_true)):
        true_label = y_true.iloc[i]
        pred_labels = list(top_k_predictions[i])
        if true_label in pred_labels:
            rank = pred_labels.index(true_label) + 1  # 排名从1开始
            map_score += 1 / rank

    return map_score / len(y_true)

# 手动计算 MAP@3
map3 = calculate_map_k(y_val, y_pred_proba, k=3)
print(f"Validation MAP@3 Score: {map3:.4f}")

# 4. 预测 Top3 肥料类型
probs = model.predict_proba(X_test)

# 获取概率最高的前3个类别索引（从高到低排序）
top3_indices = np.argsort(probs, axis=1)[:, -3:][:, ::-1]

# 解码为原始肥料名称
decode = np.vectorize(lambda x: le_label.inverse_transform([x])[0])
top3_labels = decode(top3_indices)

# 5. 生成提交文件
submission = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': [" ".join(row) for row in top3_labels]
})
submission.to_csv('submission.csv', index=False)

print("✅ 提交文件已生成：submission.csv")

