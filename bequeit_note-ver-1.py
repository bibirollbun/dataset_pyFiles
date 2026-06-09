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


# 1. Basic Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# 2. Load Data
train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
submission_format = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")

# 3. Fix Column Typo
train.rename(columns={'Temparature': 'Temperature'}, inplace=True)
test.rename(columns={'Temparature': 'Temperature'}, inplace=True)

# 4. Encode Categorical Columns
cat_cols = [col for col in train.columns if train[col].dtype == 'object' and col != 'Fertilizer Name']
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

# 5. Encode Target Label
target_le = LabelEncoder()
train["Fertilizer Name"] = target_le.fit_transform(train["Fertilizer Name"])

# 6. Prepare Features and Labels
X = train.drop(columns=["id", "Fertilizer Name"])
y = train["Fertilizer Name"]
X_test = test.drop(columns=["id"])

# 7. Split for Validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 8. Define Model with Proper Params
model = XGBClassifier(
    objective='multi:softprob',
    num_class=len(np.unique(y)),
    max_depth=10,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.6,
    colsample_bynode=0.6,
    max_bin=256,
    tree_method='gpu_hist',  # âœ… Correct GPU method
    random_state=42,
    eval_metric='mlogloss',
    n_estimators=12000,
    verbosity=0
)

# 9. Fit Model with Early Stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=100,
    verbose=100
)

# 10. Evaluate Using MAP@3 (Optional but Useful)
def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = p[:k]
        score, hits = 0.0, 0
        for i, pred in enumerate(p):
            if pred in a:
                hits += 1
                score += hits / (i + 1.0)
        return score / min(len(a), k)
    return np.mean([apk([a], p, k) for a, p in zip(actual, predicted)])

val_probs = model.predict_proba(X_val)
val_top3 = np.argsort(val_probs, axis=1)[:, -3:][:, ::-1]
map3_score = mapk(y_val, val_top3)
print(f"ðŸ“Š Validation MAP@3 Score: {map3_score:.5f}")

# 11. Predict for Test Set
test_probs = model.predict_proba(X_test)
top_3_preds = np.argsort(test_probs, axis=1)[:, -3:][:, ::-1]
top_3_labels = target_le.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)

# 12. Prepare Submission
submission = pd.DataFrame({
    'id': submission_format['id'],
    'Fertilizer Name': [' '.join(row) for row in top_3_labels]
})
submission.to_csv('submission.csv', index=False)
print("âœ… Submission saved as 'submission.csv'")


# 1. Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

# 2. Load Data
train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
submission_format = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")

# 3. Fix column typo
train.rename(columns={'Temparature': 'Temperature'}, inplace=True)
test.rename(columns={'Temparature': 'Temperature'}, inplace=True)

# 4. Encode categorical columns
cat_cols = [col for col in train.columns if train[col].dtype == 'object' and col != 'Fertilizer Name']

for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

# 5. Add interaction feature: Soil + Crop Type
# 5. Add interaction feature: Soil Type + Crop Type
train["Soil_Crop"] = train["Soil Type"].astype(str) + "_" + train["Crop Type"].astype(str)
test["Soil_Crop"] = test["Soil Type"].astype(str) + "_" + test["Crop Type"].astype(str)


le_interact = LabelEncoder()
train["Soil_Crop"] = le_interact.fit_transform(train["Soil_Crop"])
test["Soil_Crop"] = le_interact.transform(test["Soil_Crop"])

# 6. Encode target label
target_le = LabelEncoder()
train["Fertilizer Name"] = target_le.fit_transform(train["Fertilizer Name"])

# 7. Features and labels
X = train.drop(columns=["id", "Fertilizer Name"])
y = train["Fertilizer Name"]
X_test = test.drop(columns=["id"])

# 8. Define MAP@3
def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = p[:k]
        score, hits = 0.0, 0
        for i, pred in enumerate(p):
            if pred in a:
                hits += 1
                score += hits / (i + 1.0)
        return score / min(len(a), k)
    return np.mean([apk([a], p, k) for a, p in zip(actual, predicted)])

# 9. Stratified K-Fold Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros((X_test.shape[0], y.nunique()))
val_map3_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nðŸ§ª Fold {fold + 1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = XGBClassifier(
        objective='multi:softprob',
        num_class=y.nunique(),
        max_depth=10,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.6,
        colsample_bynode=0.6,
        max_bin=256,
        tree_method='gpu_hist',
        random_state=fold,
        eval_metric='mlogloss',
        n_estimators=12000,
        verbosity=0
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=100,
        verbose=100
    )
    
    # Validation MAP@3
    val_probs = model.predict_proba(X_val)
    val_top3 = np.argsort(val_probs, axis=1)[:, -3:][:, ::-1]
    map3 = mapk(y_val, val_top3)
    val_map3_scores.append(map3)
    print(f"âœ… Fold {fold + 1} MAP@3: {map3:.5f}")
    
    # Test prediction
    test_preds += model.predict_proba(X_test) / skf.n_splits

print(f"\nðŸŽ¯ Average CV MAP@3: {np.mean(val_map3_scores):.5f}")

# 10. Final Submission
top_3_preds = np.argsort(test_preds, axis=1)[:, -3:][:, ::-1]
top_3_labels = target_le.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)

submission = pd.DataFrame({
    'id': submission_format['id'],
    'Fertilizer Name': [' '.join(row) for row in top_3_labels]
})
submission.to_csv("submission.csv", index=False)
print("âœ… Submission saved as 'submission.csv'")


print("Train columns:", train.columns.tolist())
print("Test columns:", test.columns.tolist())

