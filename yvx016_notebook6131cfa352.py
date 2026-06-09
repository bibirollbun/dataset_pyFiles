#2024423320117 欧益泉

import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 1. 加载数据
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

# 2. 编码类别特征：'Soil Type', 'Crop Type'
soil_encoder = LabelEncoder()
crop_encoder = LabelEncoder()
fertilizer_encoder = LabelEncoder()

train['Soil Type'] = soil_encoder.fit_transform(train['Soil Type'])
test['Soil Type'] = soil_encoder.transform(test['Soil Type'])

train['Crop Type'] = crop_encoder.fit_transform(train['Crop Type'])
test['Crop Type'] = crop_encoder.transform(test['Crop Type'])

train['Fertilizer Name'] = fertilizer_encoder.fit_transform(train['Fertilizer Name'])

# 3. 特征和标签
X = train.drop(['id', 'Fertilizer Name'], axis=1)
y = train['Fertilizer Name']
X_test = test.drop(['id'], axis=1)

# 4. 模型训练
model = lgb.LGBMClassifier(
    objective='multiclass',
    metric='multi_logloss',
    n_estimators=500,
    learning_rate=0.05,
    num_class=len(np.unique(y))
)
model.fit(X, y)

# 5. Top5 预测
top5_pred_indices = np.argsort(model.predict_proba(X_test), axis=1)[:, -5:][:, ::-1]
top5_pred_labels = np.array([fertilizer_encoder.inverse_transform(row) for row in top5_pred_indices])

# 6. 保存结果
output = pd.DataFrame(top5_pred_labels, columns=[f'Top{i+1}' for i in range(5)])
output['id'] = test['id']
output = output[['id', 'Top1', 'Top2', 'Top3', 'Top4', 'Top5']]
output.to_csv('/kaggle/working/top5_predictions.csv', index=False)

print("✅ Top5预测结果已生成并保存为 top5_predictions.csv")

