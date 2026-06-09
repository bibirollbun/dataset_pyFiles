import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')
train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


train.info()


numerical_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
cat_cols = ['job','marital','education','default','housing','loan','contact','month','poutcome']
target_col = 'y'
id_col = 'id'


# preprocessing functions
def clip_outliers(df):
    for col in numerical_cols:
        low = df[col].quantile(0.005)
        high = df[col].quantile(0.995)
        df[col] = df[col].clip(lower=low, upper=high)
    return df

def add_log_features(df):
    df['duration_log'] = np.log(df['duration'] + 1)
    df['campaign_log'] = np.log(df['campaign'] + 1)
    df['pdays_log'] = np.log(df['pdays'] + 2)
    df['previous_log'] = np.log(df['previous'] + 1)
    return df


# preprocessing data
train = clip_outliers(train)
test = clip_outliers(test)

train = add_log_features(train)
test = add_log_features(test)


# train-validation split


X = train.drop(columns=[id_col, target_col])
y = train[target_col]
X_test = test.drop(columns=[id_col])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Categorical column indexes for CatBoost
cat_features = [X_train.columns.get_loc(col) for col in cat_cols]


model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='Logloss',
    eval_metric='AUC',
    cat_features=cat_features,
    random_seed=42,
    verbose=100,
    early_stopping_rounds=50
)

model.fit(X_train, y_train,
          eval_set=(X_val, y_val),
          use_best_model=True)


y_val_pred = model.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, y_val_pred)
print(f"Validation ROC AUC: {roc_auc:.4f}")


# predicting and saving submission
y_test_pred = model.predict_proba(X_test)[:, 1]
submission['y'] = y_test_pred
submission.to_csv("submission.csv", index=False)

