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


import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats



train_file = "/kaggle/input/detecting-reversal-points-in-us-equities/competition_data/competition_data/train.csv"
test_file = "/kaggle/input/detecting-reversal-points-in-us-equities/competition_data/competition_data/test.csv"
submission_file = "/kaggle/input/detecting-reversal-points-in-us-equities/competition_data/competition_data/sample_submission.csv"


train = pd.read_csv(train_file, low_memory=False)
test = pd.read_csv(test_file, low_memory=False)
submission = pd.read_csv(submission_file, low_memory=False)



train


test


submission


# Missing value check
print("\nMissing values:")
print("Total mission values:", train.isna().sum().sum())
print(train.isna().sum().sort_values(ascending=False))


print(test.isna().sum().sum())


print("\nSummary stats:")
print(train.describe())


train.columns.tolist()


train.info()


train.select_dtypes(include=['object'])


train.select_dtypes(include=['int64', 'float64'])


train.hist(bins=20, figsize=(10,8))
plt.show()


from sklearn.preprocessing import LabelEncoder


# mapping

mapping = {
    'HH': 'H',
    'LH': 'H',
    'HL': 'L',
    'LL': 'L',
    np.nan: 'N'
}


# Drop metadata columns
meta_cols = ['id', 'train_id', 'Unnamed: 0', 'ticker_id', 't', 'class_label']
features = [col for col in train.columns if col not in meta_cols]

# Apply mapping to target
y = train['class_label'].map(mapping)
y = y.fillna('N')  # In case there are NaN after mapping

print("Class distribution after mapping:\n", y.value_counts())

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Prepare features
X = train[features].copy()

print("\nEncoded class labels:")
for orig, enc in zip(le.classes_, range(len(le.classes_))):
    print(f"{orig} -> {enc}")



y


X


X.isna().sum().sum()


from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from lightgbm import LGBMClassifier


class_names = le.classes_
class_indices = np.arange(len(class_names))

weights = compute_class_weight(
    class_weight='balanced',
    classes=class_indices,
    y=y_encoded
)

print("\nClass weights:", dict(zip(class_names, weights)))

# Assign each sample a weight
sample_weights = np.array([weights[label] for label in y_encoded])



from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=False)



from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score

models = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n=========== Fold {fold+1} ===========")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

    w_train = sample_weights[train_idx]

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(class_names),
        eval_metric='mlogloss',
        learning_rate=0.03,
        n_estimators=600,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist"
    )

    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    models.append(model)

    preds = model.predict(X_val)
    print(classification_report(y_val, preds, target_names=class_names))



final_model = XGBClassifier(
    objective="multi:softprob",
    num_class=len(class_names),
    eval_metric='mlogloss',
    learning_rate=0.03,
    n_estimators=900,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist"
)

final_model.fit(X, y_encoded, sample_weight=sample_weights)



test_features = [col for col in test.columns if col not in meta_cols]
X_test = test[test_features]

test_prob = final_model.predict_proba(X_test)
test_pred = np.argmax(test_prob, axis=1)

# decode labels back
test_labels = le.inverse_transform(test_pred)

submission = pd.DataFrame({
    "id": test["id"],
    "label": test_labels
})

submission.to_csv("submission.csv", index=False)
submission.head()





