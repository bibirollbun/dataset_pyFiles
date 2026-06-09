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


# å­¦å�·: 2024423310127, å§“å��: è°¢å˜‰æ´‹

# ğŸ“¦ 0. å¯¼å…¥åº“
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ğŸ“� 1. è¯»å�–æ•°æ�®
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')

# ğŸ�¯ 2. æ‹†åˆ†ç‰¹å¾�å’Œæ ‡ç­¾
X = train.drop(['id', 'Fertilizer Name'], axis=1)
y = train['Fertilizer Name']
X_test = test.drop(['id'], axis=1)

# ğŸ§¼ 3. ç±»åˆ«å�˜é‡�ç¼–ç �
categorical_cols = X.select_dtypes(include='object').columns
print("éœ€è¦�ç¼–ç �çš„åˆ—:", categorical_cols.tolist())

for col in categorical_cols:
    le_col = LabelEncoder()
    X[col] = le_col.fit_transform(X[col])
    X_test[col] = le_col.transform(X_test[col])

# ğŸ�¯ ç¼–ç �ç›®æ ‡å�˜é‡�
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

# ğŸ§ª 4. åˆ‡åˆ†è®­ç»ƒ/éªŒè¯�é›†
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# ğŸŒ² 5. è®­ç»ƒ XGBoost æ¨¡å�‹
xgb_model = XGBClassifier(
    objective='multi:softprob',
    eval_metric='mlogloss',
    num_class=len(le_target.classes_),
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# ğŸ”® 6. é¢„æµ‹æµ‹è¯•é›†å¹¶å�– Top5
prob_xgb = xgb_model.predict_proba(X_test)
top5_preds_xgb = np.argsort(prob_xgb, axis=1)[:, -5:][:, ::-1]
top5_labels_xgb = le_target.inverse_transform(top5_preds_xgb.ravel()).reshape(top5_preds_xgb.shape)

# ğŸ§  7. è‡ªå®šä¹‰ MAP@5 è¯„ä¼°å‡½æ•°
def mapk(actual, predicted, k=5):
    return np.mean([
        sum([1 if p in a else 0 for p in predicted[i][:k]]) / k
        for i, a in enumerate([[label] for label in actual])
    ])

# ğŸ“ˆ 8. éªŒè¯�é›†è¯„ä¼° MAP@5
val_preds = xgb_model.predict_proba(X_val)
val_top5 = np.argsort(val_preds, axis=1)[:, -5:][:, ::-1]
val_top5_labels = le_target.inverse_transform(val_top5.ravel()).reshape(val_top5.shape)
map5_score = mapk(y_val, val_top5_labels)
print(f"ğŸ“Š XGBoost éªŒè¯�é›† MAP@5: {map5_score:.4f}")

# ğŸ“Š 9. ç‰¹å¾�é‡�è¦�æ€§å±•ç¤º
plt.figure(figsize=(12,6))
importance = xgb_model.feature_importances_
features = X.columns
sns.barplot(x=importance, y=features)
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.show()

# ğŸ“� 10. ç”Ÿæˆ�æ��äº¤æ–‡ä»¶
submission = pd.DataFrame()
submission['id'] = test['id']
for i in range(5):
    submission[f'Rank{i+1}'] = top5_labels_xgb[:, i]
submission.to_csv('submission.csv', index=False)
submission.head()


