import pandas as pd
import numpy as np
import os
import warnings
from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import time
import optuna
from sklearn.model_selection import train_test_split
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv", index_col='id')
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv", index_col='id')
submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')
original = pd.read_csv('/kaggle/input/bank-marketing-dataset-full/bank-full.csv',sep=';')


original['y'] = original['y'].map({'no': 0, 'yes': 1})


train = pd.concat([train, original], ignore_index=True)
train = train.drop_duplicates()


train.head()


train.info()


train['y'].value_counts()


train.describe()


train.isnull().sum()


def feature_engineering(df):
    df['pdays_previous_interaction'] = df['pdays'] * df['previous']

    # Balance and duration correlation feature
    df['balance_per_duration'] = df['balance'] / (df['duration'] + 1)  # +1 to avoid division by zero

    # Day and campaign correlation
    df['day_campaign_interaction'] = df['day'] * df['campaign']

    # Age binning
    df['age_group'] = pd.cut(df['age'], 
                             bins=[0, 30, 45, 60, 100], 
                             labels=['young', 'middle_aged', 'mature', 'senior'])
    df['age_group_encoded'] = pd.cut(df['age'], 
                                     bins=[0, 30, 45, 60, 100], 
                                     labels=[1, 2, 3, 4])

    # Balance binning
    balance_quartiles = df['balance'].quantile([0.25, 0.5, 0.75])
    df['balance_category'] = pd.cut(df['balance'],
                                    bins=[-float('inf'), balance_quartiles[0.25], 
                                          balance_quartiles[0.5], balance_quartiles[0.75], 
                                          float('inf')],
                                    labels=[1, 2, 3, 4])

    # Duration binning
    df['duration_category'] = pd.cut(df['duration'],
                                     bins=[0, 100, 300, 600, float('inf')],
                                     labels=[1, 2, 3, 4]) 

    # Polynomial features
    df['duration_squared'] = df['duration'] ** 2
    df['campaign_squared'] = df['campaign'] ** 2
    df['balance_squared'] = df['balance'] ** 2

    # Ratio and relative features
    df['campaign_per_previous'] = df['campaign'] / (df['previous'] + 1)
    df['pdays_previous_ratio'] = df['pdays'] / (df['previous'] + 1)
    df['age_balance_ratio'] = df['age'] / (abs(df['balance']) + 1)

    # Log transformations
    df['log_duration'] = np.log1p(df['duration'])
    df['log_campaign'] = np.log1p(df['campaign'])
    df['log_abs_balance'] = np.log1p(abs(df['balance']))
    df['balance_is_negative'] = (df['balance'] < 0).astype(int)

    return df


def fix_categorical(df, cat_cols):
    df = df.copy()
    for col in cat_cols:
        # Modern check for Categorical dtype
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].cat.add_categories("missing")
        df[col] = df[col].fillna("missing").astype(str)
    return df


train = feature_engineering(train)
test = feature_engineering(test)


categorical_features = train.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"Categorical Features: {categorical_features}")


train.head()


target = 'y'
X = train.drop(columns=target)
y = train[target]


X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


X = fix_categorical(X, categorical_features)
test_processed = fix_categorical(test.drop(columns=target, errors='ignore'), categorical_features)


X_train = fix_categorical(X_train, categorical_features)
X_valid = fix_categorical(X_valid, categorical_features)


cat_clf = CatBoostClassifier(
    allow_writing_files=False,
    verbose=False,
    task_type='GPU', 
    loss_function='CrossEntropy',
    use_best_model=True,
    cat_features=categorical_features, 
    n_estimators=10000,
    learning_rate=0.1,
    random_seed=42 # for reproducibility
)


N_SPLITS = 5
skfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)

test_pred = np.zeros(len(test_processed))
roc_scores = []

# 5. CORRECTED: Use the correctly processed dataframes (X, y, test_processed) in the loop.
for fold, (train_idx, test_idx) in enumerate(skfold.split(X, y), 1):
    print(f"===== Fold {fold} =====")
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    model = clone(cat_clf)
    model.fit(
        X_train, y_train, 
        eval_set=[(X_test, y_test)], 
        early_stopping_rounds=200, 
        verbose=500
    )
    
    y_pred = model.predict_proba(X_test)[:, 1]
    roc_score = roc_auc_score(y_test, y_pred)
    roc_scores.append(roc_score)

    # Predict on the processed test set and average the predictions
    test_pred += model.predict_proba(test_processed)[:, 1] / N_SPLITS
    print(f"Fold {fold} -> ROC-AUC: {roc_score:.5f}\n")

print("=" * 20)
print(f"Average Fold ROC-AUC: {np.mean(roc_scores):.5f} Â± {np.std(roc_scores):.5f}")
print("=" * 20)


submission['y'] = test_pred
submission.to_csv("submission.csv", index=False)
print("\nSample of final predictions:")
print(submission.head())

