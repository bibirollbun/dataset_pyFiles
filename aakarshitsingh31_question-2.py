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
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import optuna


train_df = pd.read_csv("/kaggle/input/predicting-euphoria-in-the-streets/train.csv")
test_df = pd.read_csv('/kaggle/input/predicting-euphoria-in-the-streets/test.csv')


train_ids = train_df['id']
test_ids = test_df['id']
train_df.drop('id', axis=1, inplace=True)
test_df.drop('id', axis=1, inplace=True)


X = train_df.drop('Y', axis=1)
y = train_df['Y'].astype(int)
X_test = test_df.copy()

def clean_invalid_values(df):

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Fill any NaNs that might have been created
    df.fillna(df.median(), inplace=True)
    return df

X = clean_invalid_values(X)
X_test = clean_invalid_values(X_test)
print("Infinite values cleaned.")


imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)



print("Data loaded and missing values imputed.")
print(f"Train data shape: {X.shape}")
print(f"Test data shape: {X_test.shape}")



import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 10))
corr_matrix = train_df.corr(numeric_only=True)

sns.heatmap(corr_matrix, 
            cmap='RdBu_r', 
            annot=False,  # Set to False if too many features, else it gets crowded
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={'shrink': 0.8},
            linewidths=0.5)

plt.title('Correlation Matrix After One-Hot Encoding', fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.show()


def create_features(df):
    
    poly_group_1 = ['x_1', 'x_2', 'x_4', 'x_6']
    poly_group_2 = ['x_15', 'x_16', 'x_17', 'x_18']

 
    df['x_1_x_5_diff'] = df['x_1'] - df['x_5']
    df['x_15_x_5_ratio'] = df['x_15'] / (df['x_5'] + 1e-6)
    df['x_1_x_2_sum'] = df['x_1'] + df['x_2']
    df['x_17_x_18_prod'] = df['x_17'] * df['x_18']

   
    poly1 = PolynomialFeatures(degree=2, include_bias=False)
    poly_data_1 = poly1.fit_transform(df[poly_group_1])
    poly_names_1 = poly1.get_feature_names_out(poly_group_1)
    poly_df_1 = pd.DataFrame(poly_data_1, columns=poly_names_1, index=df.index)

   
    poly2 = PolynomialFeatures(degree=2, include_bias=False)
    poly_data_2 = poly2.fit_transform(df[poly_group_2])
    poly_names_2 = poly2.get_feature_names_out(poly_group_2)
    poly_df_2 = pd.DataFrame(poly_data_2, columns=poly_names_2, index=df.index)

    
    df = pd.concat([df, poly_df_1, poly_df_2], axis=1)
    return df

X = create_features(X)
X_test = create_features(X_test)
print("Feature engineering complete.")













n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


oof_preds_rf = np.zeros(len(X))
test_preds_rf = np.zeros(len(X_test))


rf_params = {
    'n_estimators': 300, 
    'max_depth': 10, 
    'random_state': 42, 
    'n_jobs': -1,
}


for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"--- Fold {fold+1}/{n_splits} ---")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

   
    rf_model = RandomForestClassifier(**rf_params)
    rf_model.fit(X_train, y_train)
    

    oof_preds_rf[val_idx] = rf_model.predict_proba(X_val)[:, 1]


    test_preds_rf += rf_model.predict_proba(X_test)[:, 1] / n_splits

print("\nRandom Forest model training and prediction complete.")


rf_auc = roc_auc_score(y, oof_preds_rf)
print(f"Random Forest OOF ROC AUC Score: {rf_auc:.4f}")


submission_df = pd.DataFrame({'id': test_ids, 'Y': test_preds_rf})
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' created successfully!")
print(submission_df.head())







