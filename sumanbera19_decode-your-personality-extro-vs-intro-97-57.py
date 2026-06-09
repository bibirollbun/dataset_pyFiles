# Load requires libraries

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from catboost import CatBoostClassifier, Pool
# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)


# Load datasets

test_data = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
train_data = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
# submission = pd.read_csv("sample_submission.csv")

train_data.head()


test_data


train_data.columns


# train_data = train_data.drop(columns=["Friends_circle_size"])
test_data = test_data.drop(columns=["Friends_circle_size"])
test_data= test_data.drop(columns=["Post_frequency"])


train_data = train_data.drop(columns=["Friends_circle_size"])
train_data = train_data.drop(columns=["Post_frequency"])


# Transform numeric columns

scaler = StandardScaler()
num_cols = list(train_data.select_dtypes(exclude=['object']).columns)

train_data[num_cols] = scaler.fit_transform(train_data[num_cols])
test_data[num_cols] = scaler.transform(test_data[num_cols])


# Object datatype columns encoding

labelEncoder = LabelEncoder()
cat_cols = list(train_data.select_dtypes(include=['object']).columns.difference(['Personality']))

for col_name in cat_cols:
    train_data[col_name]=labelEncoder.fit_transform(train_data[col_name]).astype(int)
    test_data[col_name]=labelEncoder.transform(test_data[col_name]).astype(int)

train_data['Personality_encoded'] = labelEncoder.fit_transform(train_data['Personality'])


# Prepare data

X = train_data.drop(columns=["id", "Personality", "Personality_encoded"])
y = train_data["Personality_encoded"]
# X_test = test_data.drop(columns=["id"])

# combined = pd.concat([X, X_test], axis=0)

# X = combined.iloc[:len(X)].reset_index(drop=True)
# X_test = combined.iloc[len(X):].reset_index(drop=True)


# Correlation matrix
correlation_matrix = X.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.show()


test_data = test_data.drop(columns=["id"])



from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
model= lgb.LGBMClassifier(
    objective='binary',
    metric='binary_logloss',
    max_depth=3,
    learning_rate=0.1,      
    reg_lambda=1.0,          
    reg_alpha=0.0,            
    verbose=-1
)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test_data))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    model.fit(X_train, y_train)
    val_preds = model.predict(X_val)
    acc = accuracy_score(y_val, val_preds)

acc



model.fit(X, y)
val_preds = model.predict(X_val)
acc = accuracy_score(y_val, val_preds)
acc


a=model.predict(test_data)


submission=pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


submission["Personality"] = labelEncoder.inverse_transform(a)
submission.to_csv("submission.csv", index=False)
submission.head()




