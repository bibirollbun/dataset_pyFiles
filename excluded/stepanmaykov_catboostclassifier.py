import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
import warnings
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv", index_col='id')
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv", index_col='id')


numerical_features = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'day', 'month', 'poutcome']
target = 'y'


def preprocess_data(df, numerical_features, categorical_features):
    df = df.copy()

    for col in numerical_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(df[col].median(), inplace=True)

    df['balance_per_age'] = df['balance'] / (df['age'] + 1)
    df['duration_campaign_ratio'] = df['duration'] / (df['campaign'] + 1)
    df['pdays_binary'] = (df['pdays'] > -1).astype(int)

    all_numerical = numerical_features + ['balance_per_age', 'duration_campaign_ratio', 'pdays_binary']
    for col in all_numerical:
        df[col] = df[col].astype(str)

    for col in categorical_features:
        df[col].fillna('unknown', inplace=True)
        df[col] = df[col].astype(str)
    
    return df

train = preprocess_data(train, numerical_features, categorical_features)
test = preprocess_data(test, numerical_features, categorical_features)
all_features = numerical_features + categorical_features + ['balance_per_age', 'duration_campaign_ratio', 'pdays_binary']


X = train.drop(target, axis=1)
y = train["y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)


params = {
        'iterations': 12000,
        'learning_rate': 0.02,
        'depth': 4,
        'l2_leaf_reg': 1,
        'cat_features': all_features,
        'task_type': 'GPU',
        'verbose': 500,
        'early_stopping_rounds': 500,
        'random_seed': 0,
        "devices": "0:1"
}


n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold + 1}/{n_splits}")
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    model = CatBoostClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(test)[:, 1] / n_splits

roc_auc = roc_auc_score(y, oof_preds)
print(f"Mean CV ROC-AUC: {roc_auc:.5f}")


sub = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")
sub['y'] = test_preds
sub.to_csv("submission.csv", index=False)
sub.head()

