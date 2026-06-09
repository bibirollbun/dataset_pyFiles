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
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier


train_data = pd.read_csv(r"/kaggle/input/playground-series-s5e6/train.csv")
test_data = pd.read_csv(r"/kaggle/input/playground-series-s5e6/test.csv")
original_data = pd.read_csv(r"/kaggle/input/fertilizer-data/Fertilizer Prediction.csv")
data = pd.read_csv(r"/kaggle/input/playground-series-s5e6/sample_submission.csv")

print("train_data shape :",train_data.shape)
print("test_data shape :",test_data.shape)
print("original_data shape :",original_data.shape)
print("data shape :",data.shape)


train_data.head()


train_data.isna().sum().sort_values(ascending=False)


test_data.head()


test_data.isna().sum().sort_values(ascending=False)


original_data.head()


train_data = train_data.drop("id", axis=1)
test_data = test_data.drop("id", axis=1)
train_data = pd.concat([train_data, original_data], ignore_index=True)
train_data = train_data.drop_duplicates()
print("shape of the data :",train_data.shape)


# Encode target
target_le = LabelEncoder()
train_data['Fertilizer Name'] = target_le.fit_transform(train_data['Fertilizer Name'])
y = train_data['Fertilizer Name']


# Split features
X = train_data.drop(columns='Fertilizer Name')
test = test_data.copy()


# Identify columns
num_cols = X.select_dtypes(include='number').columns.tolist()
cat_cols = X.select_dtypes(include='object').columns.tolist()
len(num_cols), len(cat_cols)


parameters = {'booster': 'gbtree', 'lambda': 1.6421866549980566, 'alpha': 0.0020979968201803286, 'colsample_bytree': 0.49411876367383895, 'subsample': 0.9739907589393533, 'learning_rate': 0.2995583642129542, 'max_depth': 10, 'min_child_weight': 3}
#value: 0.3531576470589486.


params = {'booster': 'gbtree', 'lambda': 0.4852532041827346, 'alpha': 5.681002524055748, 'colsample_bytree': 0.40465381192194894, 'subsample': 0.9318477513237314, 'learning_rate': 0.2978528279037068, 'max_depth': 10, 'min_child_weight': 6}


# Preprocessor
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown='ignore'), cat_cols)
])


# Prepare CV variables
FOLDS = 20
kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
oof_preds = np.zeros((len(X), len(np.unique(y))))
test_preds_proba = np.zeros((len(test_data), len(np.unique(y))))

# CV loop
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Fit and transform preprocessing
    preprocessor.fit(X_train)
    X_train_scaled = preprocessor.transform(X_train)
    X_val_scaled = preprocessor.transform(X_val)
    test_scaled = preprocessor.transform(test)

  # Train model
    model = XGBClassifier(**params,
        objective='multi:softprob',
        num_class=len(np.unique(y)),
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        #n_estimators=100
    )
    model.fit(X_train_scaled, y_train)

    # Store OOF and test predictions
    oof_preds[val_idx] = model.predict_proba(X_val_scaled)
    test_preds_proba += model.predict_proba(test_scaled) / FOLDS


# Evaluate MAP@3
def map3_score(y_true, y_proba, k=3):
    top_k = np.argsort(y_proba, axis=1)[:, -k:][:, ::-1]
    score = 0.0
    for i in range(len(y_true)):
        if y_true[i] in top_k[i]:
            rank = np.where(top_k[i] == y_true[i])[0][0]
            score += 1 / (rank + 1)
    return score / len(y_true)

map3 = map3_score(y, oof_preds)
print(f"\n✅ CV MAP@3: {map3:.4f}")


# Format test predictions (top-3)
top_3 = np.argsort(test_preds_proba, axis=1)[:, -3:][:, ::-1]
top_3_labels = target_le.inverse_transform(top_3.ravel()).reshape(-1, 3)

submission = pd.DataFrame({
    "id": data.id,
    "Fertilizer Name": [" ".join(map(str, row)) for row in top_3_labels]
})
submission.to_csv("xgb_cv_submission.csv", index=False)
submission.head()




