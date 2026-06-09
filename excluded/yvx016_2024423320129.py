import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 计算 MAP@5 的函数
def mapk(actual, predicted, k=5):
    def apk(actual, predicted, k=5):
        if len(predicted) > k:
            predicted = predicted[:k]
        score = 0.0
        num_hits = 0.0
        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        if not actual:
            return 0.0
        return score / min(len(actual), k)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

# 读取 训练和测试 数据
train_data = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
print('1')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
print('1')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')
print('1')
# 分离特征和目标变量
# 去除对预测肥料类型可能无实际帮助的id列
X = train_data.drop(['id', 'Fertilizer Name'], axis=1)
y = train_data['Fertilizer Name']
X_test = test_data.drop('id', axis=1)
print('1')
# 数据预处理
# 对类别型特征进行编码
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(exclude=['object']).columns

# 定义预处理步骤
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# 对目标变量进行编码
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42)

# 模型训练与调优
# 定义模型
model = XGBClassifier(objective='multi:softprob', eval_metric=['mlogloss','map@5'])

# 定义参数网格进行调优
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5]
}

# 使用 GridSearchCV 进行超参数调优
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1_macro')
grid_search.fit(preprocessor.fit_transform(X_train), y_train)

best_model = grid_search.best_estimator_

# 模型评估
# 在验证集上进行评估
y_pred_val = best_model.predict(preprocessor.transform(X_val))

# 将验证集的预测结果转换为与真实标签同类型的形式
y_pred_val_labels = [le.inverse_transform([idx])[0] for idx in y_pred_val]
y_val_labels = le.inverse_transform(y_val)

# 计算和输出 MAP@5 分数
map5_score = mapk([[label] for label in y_val_labels], [[pred] for pred in y_pred_val_labels], k=5)
print(f"验证集上的 MAP@5 分数: {map5_score}")

# 预测与结果输出
# 对测试集进行预测
y_pred_proba = best_model.predict_proba(preprocessor.transform(X_test))

# 获取每个样本的前5种肥料及其概率
top5_indices = np.argsort(y_pred_proba, axis=1)[:, -5:][:, ::-1]
top5_probs = np.sort(y_pred_proba, axis=1)[:, -5:][:, ::-1]

# 将索引转换回肥料名称
top5_fertilizers = [le.inverse_transform(indices) for indices in top5_indices]

# 生成包含概率的预测结果
detailed_predictions = []
for i in range(len(top5_fertilizers)):
    sample_pred = []
    for j in range(5):
        fertilizer = top5_fertilizers[i][j]
        probability = top5_probs[i][j]
        sample_pred.append(f"{fertilizer}({probability:.4f})")
    detailed_predictions.append(" > ".join(sample_pred))

# 生成提交文件
submission = pd.DataFrame({
    'id': sample_submission['id'],
    'Fertilizer Name': [' '.join(fertilizers) for fertilizers in top5_fertilizers]
})
submission.to_csv('D:/kaggle_data/my_submission.csv', index=False)

# 打印前5个样本的详细预测结果
print("\n前5个样本的详细预测结果（肥料名称(概率)，按可能性从高到低排序）:")
for i, pred in enumerate(detailed_predictions[:5]):
    print(f"样本 {i+1}: {pred}")    

