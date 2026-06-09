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


df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
df.head()


df.info()


import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import LabelEncoder

# === Bước 1: Đọc dữ liệu
train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")  # test có thể không có 'Personality'

# === Bước 2: Mã hóa các cột phân loại (trừ 'Personality')
categorical_cols = ['Stage_fear', 'Drained_after_socializing']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))  # dùng cùng encoder
    label_encoders[col] = le

# === Bước 3: Các cột cần xử lý
columns_numerical = [
    'Time_spent_Alone',
    'Social_event_attendance',
    'Going_outside',
    'Friends_circle_size',
    'Post_frequency'
]

columns_categorical = [
    'Stage_fear',
    'Drained_after_socializing'
]

# === Hàm xử lý 1 cột bằng XGBoost (cho cả train/test)
def impute_column(df, col, model_type='regressor'):
    train_data = df[df[col].notnull()]
    pred_data = df[df[col].isnull()]
    if pred_data.empty:
        return df

    # Xóa các cột không nên có trong X (id, Personality, chính nó)
    X_train = train_data.drop(columns=[col, 'id'] + (['Personality'] if 'Personality' in train_data else []))
    y_train = train_data[col]
    X_pred = pred_data.drop(columns=[col, 'id'] + (['Personality'] if 'Personality' in pred_data else []))

    model = XGBRegressor(n_estimators=100, max_depth=4, random_state=42) if model_type == 'regressor' \
            else XGBClassifier(n_estimators=100, max_depth=4, random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_pred)

    df.loc[df[col].isnull(), col] = y_pred
    return df

# === Bước 4: Xử lý các cột số
for col in columns_numerical:
    train_df = impute_column(train_df, col, model_type='regressor')
    test_df = impute_column(test_df, col, model_type='regressor')

# === Bước 5: Xử lý các cột phân loại
for col in columns_categorical:
    train_df = impute_column(train_df, col, model_type='classifier')
    test_df = impute_column(test_df, col, model_type='classifier')

# === Bước 6: Kiểm tra
print("Train null values:\n", train_df.isnull().sum())
print("Test null values:\n", test_df.isnull().sum())



import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import load_iris  # ví dụ nếu bạn chưa có dữ liệu
import pandas as pd
# Tách features và label
x = train_df.drop("Personality", axis=1)
y = train_df["Personality"]

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Encode nhãn
encoder = LabelEncoder()
y = encoder.fit_transform(y)

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 2, 32, log=True)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42
    )
    score = cross_val_score(clf, x, y, cv=5, scoring='accuracy')
    return score.mean()
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
print("Best trial:")
trial = study.best_trial
print(f"  Accuracy: {trial.value}")
print("  Best hyperparameters: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")



parameter = study.best_params


from sklearn.model_selection import StratifiedKFold, cross_val_score
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model = RandomForestClassifier(**study.best_params, random_state=42)
scores = cross_val_score(model, x, y, cv=skf, scoring='accuracy')

print(f"Average CV Accuracy: {np.mean(scores):.4f}")
print(f"Standard Deviation: {np.std(scores):.4f}")


# Dự đoán trên test_df
# Chuẩn bị dữ liệu (KHÔNG chứa 'id' trong x)
x = train_df.drop(columns=["id", "Personality"])
y = train_df["Personality"]

scaler = StandardScaler()
x = scaler.fit_transform(x)

encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Chuẩn hóa test
x_test = test_df.drop(columns=["id"])
x_test = scaler.transform(x_test)

x_test = test_df.drop(columns=["id"])
x_test = scaler.transform(x_test)  # Chuẩn hóa như x_train

y_pred = model.fit(x, y).predict(x_test)  # Train trên toàn bộ dữ liệu

# Giải mã label nếu bạn đã mã hóa bằng LabelEncoder
y_pred_labels = encoder.inverse_transform(y_pred)

# Tạo file submission
submission = pd.DataFrame({
    "id": test_df["id"],
    "Personality": y_pred_labels
})
submission.to_csv("submission.csv", index=False)
print("✅ File submission.csv đã được tạo!")

submission.head()

