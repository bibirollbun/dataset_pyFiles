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


df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
original = pd.read_csv('/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv')
df = pd.concat([df, original], axis=0, ignore_index=True)
df.head()


from sklearn.preprocessing import LabelEncoder

# Encode categorical features
le_soil = LabelEncoder()
df['Soil Type'] = le_soil.fit_transform(df['Soil Type'])

le_crop = LabelEncoder()
df['Crop Type'] = le_crop.fit_transform(df['Crop Type'])

le_fert = LabelEncoder()
df['Fertilizer Name'] = le_fert.fit_transform(df['Fertilizer Name'])

# Separate features and target
X = df.drop(['id', 'Fertilizer Name'], axis=1)
y = df['Fertilizer Name']


from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import numpy as np

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    colsample_bytree=0.8,
    subsample=0.8,
    random_state=42
)

model.fit(X_train, y_train, 
          eval_set=[(X_val, y_val)])


# Predict top 3 probabilities
y_proba = model.predict_proba(X_val)
top3_preds = np.argsort(y_proba, axis=1)[:, -3:][:, ::-1]


def mapk(y_true, y_pred, k=3):
    score = 0.0
    for true, pred in zip(y_true, y_pred):
        try:
            score += 1 / (pred.index(true) + 1)
        except ValueError:
            pass
    return score / len(y_true)

# Decode predicted labels
top3_labels = [[le_fert.inverse_transform([p])[0] for p in preds] for preds in top3_preds]
true_labels = le_fert.inverse_transform(y_val)

# Evaluate
score = mapk(true_labels, top3_labels, k=3)
print(f"MAP@3 score: {score:.4f}")



test['Soil Type'] = le_soil.transform(test['Soil Type'])
test['Crop Type'] = le_crop.transform(test['Crop Type'])

X_test = test.drop(['id'], axis=1)
test_proba = model.predict_proba(X_test)
top3 = np.argsort(test_proba, axis=1)[:, -3:][:, ::-1]

# Decode predictions
pred_labels = [' '.join(le_fert.inverse_transform(row)) for row in top3]
submission = pd.DataFrame({'id': test['id'], 'Fertilizer Name': pred_labels})
submission.to_csv('submission.csv', index=False)
submission.head()




