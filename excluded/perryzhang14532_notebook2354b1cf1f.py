#2024423320123å�´å‡¯å‡¡

import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 1. åŠ è½½æ•°æ�®
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

# 2. ç¼–ç �ç±»åˆ«ç‰¹å¾�ï¼š'Soil Type', 'Crop Type'
soil_encoder = LabelEncoder()
crop_encoder = LabelEncoder()
fertilizer_encoder = LabelEncoder()

# 2.1. å¯¹ç±»åˆ«ç‰¹å¾�è¿›è¡Œç¼–ç �
train['Soil Type'] = soil_encoder.fit_transform(train['Soil Type'])
test['Soil Type'] = soil_encoder.transform(test['Soil Type'])

train['Crop Type'] = crop_encoder.fit_transform(train['Crop Type'])
test['Crop Type'] = crop_encoder.transform(test['Crop Type'])

# 2.2. ç¼–ç �ç›®æ ‡å�˜é‡� 'Fertilizer Name'
train['Fertilizer Name'] = fertilizer_encoder.fit_transform(train['Fertilizer Name'])

# 3. ç‰¹å¾�å’Œæ ‡ç­¾
X = train.drop(['id', 'Fertilizer Name'], axis=1)
y = train['Fertilizer Name']
X_test = test.drop(['id'], axis=1)

# 4. æ¨¡å�‹è®­ç»ƒ
model = lgb.LGBMClassifier(
    objective='multiclass',
    metric='multi_logloss',
    n_estimators=500,
    learning_rate=0.05,
    num_class=len(np.unique(y))
)

# 4.1. è®­ç»ƒæ¨¡å�‹
model.fit(X, y)

# 5. è¿›è¡ŒTop5é¢„æµ‹
top5_pred_indices = np.argsort(model.predict_proba(X_test), axis=1)[:, -5:][:, ::-1]
top5_pred_labels = np.array([fertilizer_encoder.inverse_transform(row) for row in top5_pred_indices])

# 6. æ�„å»ºç»“æ�œå¹¶ä¿�å­˜
output = pd.DataFrame(top5_pred_labels, columns=[f'Top{i+1}' for i in range(5)])
output['id'] = test['id']
output = output[['id', 'Top1', 'Top2', 'Top3', 'Top4', 'Top5']]

# 7. ä¿�å­˜åˆ°æ–‡ä»¶
output.to_csv('/kaggle/working/top5_predictions.csv', index=False)

print("âœ… Top5é¢„æµ‹ç»“æ�œå·²ç”Ÿæˆ�å¹¶ä¿�å­˜ä¸º top5_predictions.csv")

