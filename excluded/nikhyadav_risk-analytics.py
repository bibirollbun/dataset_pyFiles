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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb



train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
categorical_features = [
    'Marital Status', 'Education Level', 'Occupation',
    'Location', 'Policy Type', 'Customer Feedback',
    'Smoking Status', 'Exercise Frequency', 'Property Type'
]

numerical_features = [
    'Age', 'Annual Income', 'Number of Children', 'Health Score',
    'Previous Claims', 'Vehicle Age', 'Credit Score', 'Insurance Duration'
]

date_feature = 'Policy Start Date'



def handle_missing_values(df):
    df = df.copy()
    for col in numerical_features:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    for col in categorical_features:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    if date_feature in df.columns and df[date_feature].isnull().sum() > 0:
        df[date_feature].fillna(df[date_feature].mode()[0], inplace=True)
    return df

def extract_date_features(df, date_col='Policy Start Date'):
    df = df.copy()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df['Policy_Year'] = df[date_col].dt.year
        df['Policy_Month'] = df[date_col].dt.month
        df['Policy_Quarter'] = df[date_col].dt.quarter
        df['Policy_DayOfYear'] = df[date_col].dt.dayofyear
        df['Policy_DayOfWeek'] = df[date_col].dt.dayofweek
        reference_date = pd.Timestamp('2025-10-05')
        df['Policy_Age_Days'] = (reference_date - df[date_col]).dt.days
        df.drop(columns=[date_col], inplace=True)
    return df

def create_interaction_features(df):
    df = df.copy()
    if 'Age' in df.columns and 'Smoking Status' in df.columns:
        df['Age_x_Smoking'] = df['Age'] * (df['Smoking Status'] == 'Yes').astype(int)
    if 'Health Score' in df.columns and 'Exercise Frequency' in df.columns:
        df['Health_x_Exercise'] = df['Health Score'] * df['Exercise Frequency'].map({
            'Daily': 4, 'Weekly': 3, 'Monthly': 2, 'Rarely': 1, 'Never': 0
        }).fillna(0)
    if 'Annual Income' in df.columns and 'Number of Children' in df.columns:
        df['Income_per_Child'] = df['Annual Income'] / (df['Number of Children'] + 1)
    if 'Credit Score' in df.columns and 'Annual Income' in df.columns:
        df['Credit_to_Income'] = df['Credit Score'] / (df['Annual Income'] + 1)
    if 'Previous Claims' in df.columns and 'Insurance Duration' in df.columns:
        df['Claims_per_Year'] = df['Previous Claims'] / (df['Insurance Duration'] + 0.1)
    if 'Vehicle Age' in df.columns:
        df['Vehicle_Age_Squared'] = df['Vehicle Age'] ** 2
    if 'Age' in df.columns:
        df['Age_Group'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 55, 65, 100],
                                 labels=[1, 2, 3, 4, 5, 6], right=False)
        df['Age_Group'] = df['Age_Group'].astype(float).fillna(0).astype(int)
    return df



train_df = handle_missing_values(train_df)
test_df = handle_missing_values(test_df)
train_df = extract_date_features(train_df)
test_df = extract_date_features(test_df)
train_df = create_interaction_features(train_df)
test_df = create_interaction_features(test_df)



label_encoders = {}
for col in categorical_features:
    if col in train_df.columns and col in test_df.columns:
        le = LabelEncoder()
        combined_values = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)
        le.fit(combined_values)
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))
        label_encoders[col] = le



label_encoders = {}
for col in categorical_features:
    if col in train_df.columns and col in test_df.columns:
        le = LabelEncoder()
        combined_values = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)
        le.fit(combined_values)
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))
        label_encoders[col] = le



X = train_df.drop(columns=['id', 'Premium Amount'])
y = train_df['Premium Amount']
X_test = test_df.drop(columns=['id'])
test_ids = test_df['id'].values



THRESHOLD = 1.0
low_premium_mask = y <= THRESHOLD
high_premium_mask = y > THRESHOLD
y_class = (y > THRESHOLD).astype(int)

def create_target_bins(y, n_bins=10):
    return pd.qcut(y, q=n_bins, labels=False, duplicates='drop')

y_bins = create_target_bins(y, n_bins=10)
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)



classifier_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'max_depth': 8,
    'learning_rate': 0.05,
    'min_data_in_leaf': 100,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 3,
    'lambda_l1': 0.5,
    'lambda_l2': 2.0,
    'verbose': -1,
    'random_state': 42,
    'force_col_wise': True
}

oof_class_probs = np.zeros(len(X))
test_class_probs = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_bins), 1):
    X_train_fold = X.iloc[train_idx]
    y_class_fold = y_class.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]
    y_class_val = y_class.iloc[val_idx]

    train_data = lgb.Dataset(X_train_fold, label=y_class_fold, categorical_feature=categorical_features)
    val_data = lgb.Dataset(X_val_fold, label=y_class_val, categorical_feature=categorical_features, reference=train_data)

    clf = lgb.train(
        classifier_params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    oof_class_probs[val_idx] = clf.predict(X_val_fold, num_iteration=clf.best_iteration)
    test_class_probs += clf.predict(X_test, num_iteration=clf.best_iteration) / n_folds

oof_class_pred = (oof_class_probs > 0.5).astype(int)
test_class_pred = (test_class_probs > 0.5).astype(int)
accuracy = accuracy_score(y_class, oof_class_pred)



X_low = X[low_premium_mask].reset_index(drop=True)
y_low = y[low_premium_mask].reset_index(drop=True)
y_low_bins = create_target_bins(y_low, n_bins=5)
skf_low = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

params_low = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 100,
    'max_depth': 10,
    'learning_rate': 0.03,
    'min_data_in_leaf': 50,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.85,
    'bagging_freq': 5,
    'lambda_l1': 0.3,
    'lambda_l2': 1.5,
    'min_gain_to_split': 0.01,
    'verbose': -1,
    'random_state': 42,
    'force_col_wise': True
}

oof_low_pred = np.zeros(len(y_low))
test_low_pred = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf_low.split(X_low, y_low_bins), 1):
    X_train_fold = X_low.iloc[train_idx]
    y_train_fold = y_low.iloc[train_idx]
    X_val_fold = X_low.iloc[val_idx]
    y_val_fold = y_low.iloc[val_idx]

    train_data = lgb.Dataset(X_train_fold, label=y_train_fold, categorical_feature=categorical_features)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, categorical_feature=categorical_features, reference=train_data)

    model = lgb.train(
        params_low,
        train_data,
        num_boost_round=3000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )

    oof_low_pred[val_idx] = model.predict(X_val_fold, num_iteration=model.best_iteration)
    test_low_pred += model.predict(X_test, num_iteration=model.best_iteration) / 5

low_rmse = np.sqrt(mean_squared_error(y_low, oof_low_pred))



X_high = X[high_premium_mask].reset_index(drop=True)
y_high = y[high_premium_mask].reset_index(drop=True)
y_high_bins = create_target_bins(y_high, n_bins=5)
skf_high = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

params_high = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 100,
    'max_depth': 10,
    'learning_rate': 0.03,
    'min_data_in_leaf': 50,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.85,
    'bagging_freq': 5,
    'lambda_l1': 0.3,
    'lambda_l2': 1.5,
    'min_gain_to_split': 0.01,
    'verbose': -1,
    'random_state': 123,
    'force_col_wise': True
}

oof_high_pred = np.zeros(len(y_high))
test_high_pred = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf_high.split(X_high, y_high_bins), 1):
    X_train_fold = X_high.iloc[train_idx]
    y_train_fold = y_high.iloc[train_idx]
    X_val_fold = X_high.iloc[val_idx]
    y_val_fold = y_high.iloc[val_idx]

    train_data = lgb.Dataset(X_train_fold, label=y_train_fold, categorical_feature=categorical_features)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, categorical_feature=categorical_features, reference=train_data)

    model = lgb.train(
        params_high,
        train_data,
        num_boost_round=3000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )

    oof_high_pred[val_idx] = model.predict(X_val_fold, num_iteration=model.best_iteration)
    test_high_pred += model.predict(X_test, num_iteration=model.best_iteration) / 5

high_rmse = np.sqrt(mean_squared_error(y_high, oof_high_pred))


