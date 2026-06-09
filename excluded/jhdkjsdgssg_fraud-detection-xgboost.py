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


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# Step 1: 讀檔
trans = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_transaction.csv")
iden = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_identity.csv")

# Step 2: 合併資料
df = trans.merge(iden, how='left', on='TransactionID')

# Step 3: 簡單處理缺值（先用 -999 填）
df.fillna(-999, inplace=True)


# Step 4: 把 label 拿出來
y = df['isFraud']
X = df.drop(['isFraud', 'TransactionID'], axis=1)

# Step 5: 只保留數值欄位（因為還沒做 encoding）
X = X.select_dtypes(include=[np.number])

# Step 6: 分訓練 / 驗證集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

# Step 7: 建立 XGBoost 模型
model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, use_label_encoder=False, eval_metric='auc')
model.fit(X_train, y_train)

# Step 8: 預測並評估
y_pred = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred)
print(f"AUC: {auc:.4f}")


