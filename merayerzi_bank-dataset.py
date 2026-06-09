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


# 1. Verileri oku
train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


train.head()


test.head()


from sklearn.preprocessing import LabelEncoder

# 2. Hedef deÄŸiÅŸken ve test id'si
y_train = train['y']
test_ids = test['id']

# 3. Hedefi Ã§Ä±kar, 'id' sÃ¼tunlarÄ±nÄ± da at
train = train.drop(['y', 'id'], axis=1)
test = test.drop(['id'], axis=1)

# 4. BirleÅŸtir (aynÄ± iÅŸlemi tÃ¼m veriye uygulamak iÃ§in)
full = pd.concat([train, test], axis=0).reset_index(drop=True)


full.head()


full.shape


full.info()


full.describe().T


# 5. Kategorik sÃ¼tunlar
cat_cols = full.select_dtypes(include='object').columns

# 6. 'unknown' -> NaN
for col in cat_cols:
    full[col] = full[col].replace('unknown', np.nan)

# 7. Eksikleri mod ile doldur
for col in cat_cols:
    mode_val = full[col].mode()[0]
    full[col] = full[col].fillna(mode_val)

# 8. Label Encoding
for col in cat_cols:
    le = LabelEncoder()
    full[col] = le.fit_transform(full[col].astype(str))

# 9. train/test tekrar ayÄ±r
X_train = full.iloc[:len(y_train), :]
X_test = full.iloc[len(y_train):, :]

# Kontrol
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)


# y'nin sÄ±nÄ±f daÄŸÄ±lÄ±mÄ±
print("SÄ±nÄ±f sayÄ±larÄ±:\n", y_train.value_counts())
print("\nSÄ±nÄ±f oranlarÄ± (%):\n", y_train.value_counts(normalize=True) * 100)


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

# TÃ¼m eÄŸitim verisi
X_full = X_train  # daha Ã¶nce oluÅŸturmuÅŸtuk (encoding vs. sonrasÄ±)
y_full = y_train

# DoÄŸrulama seti ayÄ±r (stratified)
X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)

# Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'  # ğŸ‘ˆ burasÄ± eklendi unbalanced data olduÄŸu iÃ§in
)
model.fit(X_train, y_train)

# Tahminler
y_pred = model.predict(X_val)
y_proba = model.predict_proba(X_val)[:, 1]

# Metrikler
f1 = f1_score(y_val, y_pred)
roc_auc = roc_auc_score(y_val, y_proba)
ap = average_precision_score(y_val, y_proba)

print("F1 Score:", round(f1, 4))
print("ROC AUC:", round(roc_auc, 4))
print("Average Precision (AP):", round(ap, 4))



X_kaggle_test = X_test  # daha Ã¶nce ayÄ±rdÄ±ÄŸÄ±mÄ±z test verisi



y_kaggle_preds = model.predict(X_kaggle_test)
submission = pd.DataFrame({"id": test_ids, "y": y_kaggle_preds})
submission.to_csv("submission.csv", index=False)


