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


# 学号: 2024423310127, 姓名: 谢嘉洋

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 读取数据
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
sample = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')

# 2. 目标变量编码
le = LabelEncoder()
train['Fertilizer Name'] = le.fit_transform(train['Fertilizer Name'])

# 3. 合并处理训练集与测试集的分类特征
combined = pd.concat([train.drop('Fertilizer Name', axis=1), test], axis=0)

# 4. Label Encoding 分类特征
for col in ['Soil Type', 'Crop Type']:
    le_col = LabelEncoder()
    combined[col] = le_col.fit_transform(combined[col])

# 拆分回训练与测试集
X = combined.iloc[:len(train), :]
X_test = combined.iloc[len(train):, :]
y = train['Fertilizer Name']

# 5. 训练模型
model = LGBMClassifier(
    objective='multiclass',
    num_class=len(le.classes_),
    metric='multi_logloss',
    n_estimators=300,
    learning_rate=0.05,
    random_state=42
)
model.fit(X, y)

# 6. Top5 预测
proba = model.predict_proba(X_test)
top5 = np.argsort(proba, axis=1)[:, -5:][:, ::-1]

# 7. 构造提交文件
submission = pd.DataFrame({
    'id': test['id'],
    'Predicted': [' '.join(le.inverse_transform(row)) for row in top5]
})
submission.to_csv('submission.csv', index=False)

# 8. 可视化特征重要性
plt.figure(figsize=(10, 6))
sns.barplot(
    x=model.feature_importances_,
    y=X.columns
)
plt.title('Feature Importance')
plt.show()


