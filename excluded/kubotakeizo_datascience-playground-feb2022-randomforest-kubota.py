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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


train_df = pd.read_csv('../input/tabular-playground-series-feb-2022/train.csv')
test_df = pd.read_csv('../input/tabular-playground-series-feb-2022/test.csv')
sample_submission = pd.read_csv('../input/tabular-playground-series-feb-2022/sample_submission.csv')


print(f"trainデータ: {train_df.shape}")
print(f"testデータ: {test_df.shape}")


le = LabelEncoder()
train_df['target_num'] = le.fit_transform(train_df['target'])
target_names = le.classes_

for i, name in enumerate(le.classes_):
    print(f"{i}: {name}")


features = [col for col in train_df.columns if col not in ['row_id', 'target', 'target_num']]

X_full = train_df[features]
y_full = train_df['target_num']
X_test = test_df[features]

print(f"使用する特徴量の数: {len(features)}")


X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

print(f"学習用データ: {X_train.shape}")
print(f"検証用データ: {X_val.shape}")


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# model = LogisticRegression(max_iter=100000, random_state=42)
# model.fit(X_train_scaled, y_train)

model = RandomForestClassifier(n_estimators=700, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

# model = XGBClassifier(n_estimators=500, learning_rate=0.05, random_state=42, use_label_encoder=False, eval_metric='mlogloss', tree_method='hist', device='cuda')
# model.fit(X_train_scaled, y_train)


val_predictions = model.predict(X_val_scaled)
accuracy = accuracy_score(y_val, val_predictions)

print(f"検証データでの正解率 (Accuracy): {accuracy:.4f}")


predictions_num = model.predict(X_test_scaled)
predictions_target = le.inverse_transform(predictions_num)


submission_df = pd.DataFrame({'row_id': test_df['row_id'], 'target': predictions_target})
submission_df.to_csv('submission.csv', index=False)
print(submission_df.head())

