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

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/dstc-kaggle-practice-2025/train.csv")
test = pd.read_csv("/kaggle/input/dstc-kaggle-practice-2025/test.csv")
sample = pd.read_csv("/kaggle/input/dstc-kaggle-practice-2025/sample_submission.csv")

print(" Data loaded successfully!")
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Sample shape:", sample.shape)


train_cols = set(train.columns)
test_cols = set(test.columns)
possible_targets = list(train_cols - test_cols)

if len(possible_targets) == 0:
    raise ValueError("â�Œ Tidak ditemukan kolom target otomatis, pastikan kolom target tidak ada di test.csv!")

target_col = possible_targets[0]
print("ğŸ�¯ Kolom target terdeteksi:", target_col)


unique_vals = train[target_col].nunique()
if unique_vals == 2:
    problem_type = "binary_classification"
elif unique_vals <= 10 and pd.api.types.is_integer_dtype(train[target_col]):
    problem_type = "multiclass"
else:
    problem_type = "regression"

print("ğŸ”� Detected problem type:", problem_type)


print("\nMissing values (Top 10):")
print(train.isna().sum().sort_values(ascending=False).head(10))

plt.figure(figsize=(5,4))
if problem_type == "binary_classification":
    sns.countplot(x=target_col, data=train)
    plt.title("Distribusi Target")
else:
    sns.histplot(train[target_col], bins=30, kde=True)
    plt.title("Distribusi Target (Regresi)")
plt.show()


def basic_feature_engineering(df):
    df = df.copy()
    # Parse datetime
    for col in df.columns:
        if df[col].dtype == "object":
            if df[col].astype(str).str.contains(r'\d{4}-\d{2}-\d{2}', regex=True).any():
                df[col] = pd.to_datetime(df[col], errors='ignore')
    datetime_cols = df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns.tolist()
    for dt in datetime_cols:
        df[dt + "_year"] = df[dt].dt.year
        df[dt + "_month"] = df[dt].dt.month
        df[dt + "_day"] = df[dt].dt.day
        df[dt + "_hour"] = df[dt].dt.hour
        df[dt + "_weekday"] = df[dt].dt.weekday
    # Lokasi (optional)
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df['lat_round'] = df['latitude'].round(2)
        df['lon_round'] = df['longitude'].round(2)
        df['latlon'] = df['lat_round'].astype(str) + "_" + df['lon_round'].astype(str)
    return df

train_fe = basic_feature_engineering(train)
test_fe = basic_feature_engineering(test)


ID_COLS = ['id', 'Id', 'ID']
id_col = [c for c in train_fe.columns if c.lower() in [x.lower() for x in ID_COLS]]
id_col = id_col[0] if id_col else None

excluded = [target_col] + ([id_col] if id_col else [])
features = [c for c in train_fe.columns if c not in excluded]

num_features = train_fe[features].select_dtypes(include=[np.number]).columns.tolist()
cat_features = train_fe[features].select_dtypes(include=['object', 'category']).columns.tolist()

print("Numerical features:", len(num_features))
print("Categorical features:", len(cat_features))


num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

low_card = [c for c in cat_features if train_fe[c].nunique() <= 20]
high_card = [c for c in cat_features if train_fe[c].nunique() > 20]

cat_low_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

cat_high_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipe, num_features),
    ('cat_low', cat_low_pipe, low_card),
    ('cat_high', cat_high_pipe, high_card)
], remainder='drop')


X = train_fe[features]
y = train_fe[target_col]

if problem_type == "binary_classification":
    model = LogisticRegression(max_iter=1000)
elif problem_type == "multiclass":
    model = LogisticRegression(max_iter=1000, multi_class='multinomial')
else:
    model = RandomForestRegressor(n_estimators=100, random_state=42)

pipe = Pipeline([
    ('preproc', preprocessor),
    ('model', model)
])

if problem_type.startswith("binary") or problem_type == "multiclass":
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
else:
    splitter = KFold(n_splits=5, shuffle=True, random_state=42)

scores = []
for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_val)
    
    if problem_type == "binary_classification":
        if hasattr(pipe, "predict_proba"):
            score = roc_auc_score(y_val, pipe.predict_proba(X_val)[:, 1])
        else:
            score = roc_auc_score(y_val, preds)
        print(f"Fold {fold} AUC: {score:.4f}")
    elif problem_type == "multiclass":
        score = accuracy_score(y_val, preds)
        print(f"Fold {fold} ACC: {score:.4f}")
    else:
        score = mean_squared_error(y_val, preds, squared=False)
        print(f"Fold {fold} RMSE: {score:.4f}")
    scores.append(score)

print(" Mean CV score:", np.mean(scores))


if problem_type == "binary_classification":
    final_model = LogisticRegression(max_iter=1000)
elif problem_type == "multiclass":
    final_model = LogisticRegression(max_iter=1000, multi_class='multinomial')
else:
    final_model = RandomForestRegressor(n_estimators=200, random_state=42)

final_pipe = Pipeline([
    ('preproc', preprocessor),
    ('model', final_model)
])

final_pipe.fit(X, y)
test_preds = final_pipe.predict(test_fe[features])

print(" Prediction done. Sample preds:", test_preds[:5])


submission = sample.copy()

target_cols = [c for c in submission.columns if c != id_col]
if len(target_cols) == 1:
    submission[target_cols[0]] = test_preds
else:
    if problem_type == "binary_classification":
        if hasattr(final_pipe, "predict_proba"):
            proba = final_pipe.predict_proba(test_fe[features])[:, 1]
            submission[target_cols[0]] = proba
    elif problem_type == "multiclass":
        probs = final_pipe.predict_proba(test_fe[features])
        for i, col in enumerate(target_cols):
            submission[col] = probs[:, i]
    else:
        submission[target_cols[0]] = test_preds

submission.to_csv("submission.csv", index=False)
print("ğŸ’¾ submission.csv saved successfully!")

