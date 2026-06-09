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
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os

# 自定义MAP@5评估函数
def map5(y_true, y_pred):
    y_pred = y_pred.argsort(axis=1)[:, ::-1][:, :5]
    score = 0.0
    for i in range(y_true.shape[0]):
        for k in range(min(5, y_pred.shape[1])):
            if y_pred[i, k] == y_true[i]:
                score += 1.0 / (k + 1)
                break
    return score / y_true.shape[0]

# 数据加载和预处理
try:
    # 检查文件是否存在
    train_path = 'train.csv'
    test_path = 'test.csv'
    
    # 如果文件不在当前目录，尝试从Kaggle输入目录查找
    if not os.path.exists(train_path):
        train_path = '/kaggle/input/playground-series-s5e6/train.csv'
    if not os.path.exists(test_path):
        test_path = '/kaggle/input/playground-series-s5e6/test.csv'
    
    # 加载数据
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    print("数据加载成功！")
    print(f"训练集形状: {train_data.shape}")
    print(f"测试集形状: {test_data.shape}")

    # 打印列名帮助调试
    print("\n训练集列名:", train_data.columns.tolist())
    print("测试集列名:", test_data.columns.tolist())

except FileNotFoundError as e:
    print(f"错误: 无法找到数据文件 - {e}")
    print("请确保文件路径正确，或者文件已上传到正确目录")
    print("\n当前目录文件列表:")
    print(os.listdir())
    raise

except Exception as e:
    print(f"加载数据时发生错误: {e}")
    raise

# 根据截图中的列名调整特征和目标列
features = [
    'Nitrogen',        # 对应N
    'Phosphorous',     # 对应P
    'Potassium',       # 对应K
    'Temparature',     # 对应temperature (注意拼写)
    'Humidity',        # 对应humidity
    'Moisture'         # 作为pH和rainfall的替代
]

target_column = 'Fertilizer Name'  # 根据截图修正

# 检查这些列是否都存在
missing_features = [col for col in features if col not in train_data.columns]
if missing_features:
    raise ValueError(f"错误: 以下特征列不存在: {missing_features}")

if target_column not in train_data.columns:
    raise ValueError(f"错误: 目标列 '{target_column}' 不存在")

# 修正此处print语句，添加右括号
print("\n使用的特征列:", features)
print("使用的目标列:", target_column)

# 准备数据
X = train_data[features]
y = train_data[target_column]

# 标签编码
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 数据集划分
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# XGBoost模型训练
model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=len(le.classes_),
    eval_metric='mlogloss',
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

print("\n开始模型训练...")
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=20,
    verbose=10
)
print("模型训练完成！")

# 预测概率
probabilities = model.predict_proba(X_val)

# 计算MAP@5
y_pred_top5 = np.argsort(-probabilities, axis=1)[:, :5]
map5_score = map5(y_val, probabilities)
print(f"\nValidation MAP@5: {map5_score:.4f}")

# 测试集预测 - 使用匹配的特征列
print("\n开始测试集预测...")
test_probabilities = model.predict_proba(test_data[features])
test_pred_top5 = np.argsort(-test_probabilities, axis=1)[:, :5]
print("测试集预测完成！")

# 转换为原始标签
top5_fertilizers = []
for row in test_pred_top5:
    top5_fertilizers.append(' '.join(le.inverse_transform(row)))

# 生成提交文件 - 修改后的部分
print("\n准备生成提交文件...")
# 检查测试数据是否有ID列，如果没有则使用索引作为ID
if 'ID' not in test_data.columns:
    print("警告: 测试数据中没有'ID'列，将使用索引作为ID")
    test_data['ID'] = test_data.index.astype(str)  # 确保ID为字符串类型
    print(f"已创建ID列，示例ID: {test_data['ID'].head().tolist()}")
else:
    print(f"使用现有的ID列，示例ID: {test_data['ID'].head().tolist()}")

# 创建提交DataFrame
submission = pd.DataFrame({
    'ID': test_data['ID'],
    'fertilizer': top5_fertilizers  # 保持输出列名为fertilizer以符合要求
})

# 打印提交文件预览
print("\n提交文件前5行预览:")
print(submission.head())

# 保存提交文件
submission.to_csv('submission.csv', index=False)
print("\n提交文件已成功生成: submission.csv")
print(f"文件大小: {os.path.getsize('submission.csv')} 字节")
print("程序执行完毕！")

