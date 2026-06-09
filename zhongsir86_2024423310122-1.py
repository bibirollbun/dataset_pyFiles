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


#陆志霖
#2024423310122
#计科1班


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
import matplotlib
import os

# ========== 1. 解决中文显示问题 ==========
# 安装中文字体（Kaggle环境）
!apt-get install fonts-wqy-zenhei -qq > /dev/null
!fc-cache -fv > /dev/null

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']  # 使用安装的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ========== 2. 数据加载与预处理 ==========
# 读取数据
train_df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

# 修正列名拼写错误
train_df = train_df.rename(columns={
    'Temparature': 'Temperature',
    'Fertilizer Name': 'fertilizer'
})
test_df = test_df.rename(columns={'Temparature': 'Temperature'})

# ========== 3. 数据可视化 ==========
# 目标变量分布
plt.figure(figsize=(12, 6))
sns.countplot(data=train_df, y='fertilizer', 
             order=train_df['fertilizer'].value_counts().index)
plt.title('肥料类型分布')
plt.tight_layout()
plt.savefig('target_distribution.png', bbox_inches='tight', dpi=300)
plt.close()

# ========== 4. 数据预处理 ==========
def preprocess_data(df, encoder=None, fit=False):
    df = df.copy()
    categorical_cols = ['Soil Type', 'Crop Type']
    
    if categorical_cols:
        if fit:
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            df[categorical_cols] = encoder.fit_transform(df[categorical_cols])
        else:
            if encoder is not None:
                df[categorical_cols] = encoder.transform(df[categorical_cols])
    
    return df, encoder

train_df, encoder = preprocess_data(train_df, fit=True)
test_df, _ = preprocess_data(test_df, encoder=encoder)

# 目标变量编码
label_encoder = LabelEncoder()
train_df['target_encoded'] = label_encoder.fit_transform(train_df['fertilizer'])

# ========== 5. 模型训练 ==========
# 准备数据
X = train_df.drop(['id', 'fertilizer', 'target_encoded'], axis=1)
y = train_df['target_encoded']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 模型参数
params = {
    'objective': 'multi:softprob',
    'num_class': len(label_encoder.classes_),
    'eval_metric': 'mlogloss',
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42,
    'tree_method': 'hist',
    'early_stopping_rounds': 50
}

# 训练模型
model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train,
          eval_set=[(X_train, y_train), (X_val, y_val)],
          verbose=50)

# ========== 6. 特征重要性分析 ==========
plt.figure(figsize=(12, 8))
xgb.plot_importance(model, max_num_features=15)
plt.title('特征重要性')
plt.tight_layout()
plt.savefig('feature_importance.png', bbox_inches='tight', dpi=300)
plt.close()

# ========== 7. 模型评估 ==========
def calculate_map5(true_labels, predicted_probs, k=5):
    map5_score = 0.0
    top5 = np.argsort(-predicted_probs, axis=1)[:, :k]
    for i in range(len(true_labels)):
        for j in range(1, k+1):
            if true_labels.iloc[i] in top5[i, :j]:
                map5_score += 1 / j
    return map5_score / len(true_labels)

val_probs = model.predict_proba(X_val)
map5_score = calculate_map5(y_val, val_probs)
print(f"Validation MAP@5: {map5_score:.4f}")

# ========== 8. 生成预测结果 ==========
# 测试集预测
test_probs = model.predict_proba(test_df.drop('id', axis=1))
top5_test = np.argsort(-test_probs, axis=1)[:, :5]

# 解码预测结果
decoded_predictions = []
for row in top5_test:
    decoded_row = label_encoder.inverse_transform(row)
    decoded_predictions.append(' '.join(decoded_row))

# 预测结果可视化
sample_idx = 0
plt.figure(figsize=(12, 6))
sns.barplot(x=label_encoder.classes_, y=test_probs[sample_idx])
plt.xticks(rotation=45)
plt.title('测试样本预测概率分布')
plt.ylabel('预测概率')
plt.tight_layout()
plt.savefig('sample_prediction.png', bbox_inches='tight', dpi=300)
plt.close()

# ========== 9. 生成提交文件 ==========
submission = pd.DataFrame({
    'id': test_df['id'],
    'fertilizer': decoded_predictions
})
submission.to_csv('submission.csv', index=False)

# ========== 10. 验证文件生成 ==========
print("\n生成的文件列表:")
print(os.listdir('/kaggle/working/'))
print("\n文件生成完成！")

