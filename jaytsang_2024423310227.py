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


# 学号: 2024423310227, 姓名: 曾俊杰

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')

categorical_cols = ['Soil Type', 'Crop Type']
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    encoders[col] = le

target = 'Fertilizer Name'
label_encoder = LabelEncoder()
train['label'] = label_encoder.fit_transform(train[target])
class_names = label_encoder.classes_

features = [col for col in train.columns if col not in ['id', target, 'label']]
X = train[features]
y = train['label']
X_test = test[features]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

model = XGBClassifier(
    objective='multi:softprob',
    num_class=len(class_names),
    n_estimators=500,
    learning_rate=0.05,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)

model.fit(X_train, y_train)

valid_probs = model.predict_proba(X_valid)
top5_valid = np.argsort(valid_probs, axis=1)[:, -5:][:, ::-1]
top5_valid_labels = np.vectorize(lambda x: class_names[x])(top5_valid)
true_labels_valid = label_encoder.inverse_transform(y_valid)

def mapk(actual, predicted, k=5):
    total = 0.0
    for a, p in zip(actual, predicted):
        if a in p[:k]:
            total += 1.0 / (p[:k].index(a) + 1)
    return total / len(actual)

map5_score = mapk(true_labels_valid, top5_valid_labels.tolist(), k=5)
print(f"\n MAP@5 on validation set: {map5_score:.4f}\n")

for i in range(10):
    print(f"样本 {i+1}")
    print(f" True Fertilizer: {true_labels_valid[i]}")
    print(f" Top-5 Predicted: {list(top5_valid_labels[i])}")
    if true_labels_valid[i] in top5_valid_labels[i]:
        rank = list(top5_valid_labels[i]).index(true_labels_valid[i]) + 1
        score = 1.0 / rank
    else:
        score = 0.0
    print(f" MAP@5 Score for this sample: {score:.3f}\n")

probs = model.predict_proba(X_test)
top5 = np.argsort(probs, axis=1)[:, -5:][:, ::-1]
top5_labels = np.vectorize(lambda x: class_names[x])(top5)
submission = pd.DataFrame({
    'id': test['id'], 
    'Fertilizer Name': [' '.join(row) for row in top5_labels]
})

submission.to_csv('submission.csv', index=False)
submission.head()


