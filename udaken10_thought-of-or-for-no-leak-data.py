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

train = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/train.csv')
test = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/test.csv')

# Drop ID and target from train
train_features = train.drop(['ID', 'HOMELESS_RATE'], axis=1)
# Drop ID from test
test_features = test.drop(['ID'], axis=1)

# Check for duplicates within train
train_dups = train_features.duplicated().sum()
print(f"Duplicates within train: {train_dups}")

# Check for duplicates within test
test_dups = test_features.duplicated().sum()
print(f"Duplicates within test: {test_dups}")

# Check for rows in test that are also in train
# We merge and check for matching rows
merged = pd.merge(train_features, test_features, how='inner')
print(f"Overlapping rows between train and test: {len(merged)}")

if len(merged) > 0:
    print("Found following overlapping rows (showing first 5 if more):")
    print(merged.head())
    
    # Let's find which IDs in train and test correspond to these overlaps
    train['features_tuple'] = train_features.apply(tuple, axis=1)
    test['features_tuple'] = test_features.apply(tuple, axis=1)
    
    overlap_tuples = merged.apply(tuple, axis=1).tolist()
    
    train_overlap_ids = train[train['features_tuple'].isin(overlap_tuples)]['ID'].tolist()
    test_overlap_ids = test[test['features_tuple'].isin(overlap_tuples)]['ID'].tolist()
    
    print(f"IDs in Train that overlap: {train_overlap_ids}")
    print(f"IDs in Test that overlap: {test_overlap_ids}")

# Check if any feature has constant value
constant_features = [col for col in train_features.columns if train_features[col].nunique() <= 1]
print(f"Constant features in Train: {constant_features}")

# Check suspicion: AGE_25_PLUS_PCT vs others
def check_age_relations(df, name):
    print(f"\n--- Age Relation checks for {name} ---")
    age_25_plus_cols = ['AGE_25_34_PCT', 'AGE_35_44_PCT', 'AGE_45_54_PCT', 'AGE_55_59_PCT', 
                        'AGE_60_61_PCT', 'AGE_62_64_PCT', 'AGE_65_69_PCT', 'AGE_70_79_PCT', 
                        'AGE_80_PLUS_PCT']
    sum_25_plus = df[age_25_plus_cols].sum(axis=1)
    diff = (sum_25_plus - df['AGE_25_PLUS_PCT']).abs()
    print(f"Mean Abs Diff (Sum(25-80+) vs AGE_25_PLUS_PCT): {diff.mean():.6f}")
    
    sum_all = df['AGE_U18_PCT'] + df['AGE_18_24_PCT'] + df['AGE_25_PLUS_PCT']
    print(f"Sum(U18 + 18-24 + 25+) (min/max): {sum_all.min():.4f} / {sum_all.max():.4f}")

    # Check for identical columns
    print("\n--- Identical Column Check ---")
    cols = df.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if (df[cols[i]] == df[cols[j]]).all():
                print(f"Col '{cols[i]}' and '{cols[j]}' are IDENTICAL")

check_age_relations(train, "Train")
check_age_relations(test, "Test")

# ID Prefix Analysis
train['Prefix'] = train['ID'].apply(lambda x: x.split('_')[0])
test['Prefix'] = test['ID'].apply(lambda x: x.split('_')[0])

prefix_stats = train.groupby('Prefix')['HOMELESS_RATE'].agg(['mean', 'count']).sort_values('mean', ascending=False)
print("\n--- HOMELESS_RATE by ID Prefix ---")
print(prefix_stats)

# ID Number Analysis
train['Number'] = train['ID'].apply(lambda x: int(x.split('_')[1]))
print(f"\n--- Correlation of ID Number with HOMELESS_RATE: {train['Number'].corr(train['HOMELESS_RATE']):.4f}")

# Find any correlations > 0.95 between features
print("\n--- Highly correlated features (> 0.95) ---")
corr_matrix = train_features.corr().abs()
high_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > 0.95:
            high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

for col1, col2, val in high_corr:
    print(f"{col1} <-> {col2}: {val:.4f}")

# Check if any feature in Train has a very high correlation with HOMELESS_RATE (> 0.8)
# high_target_corr = correlations[correlations > 0.8]
# print(f"\nFeatures highly correlated with HOMELESS_RATE (> 0.8): {high_target_corr.index.tolist()}")



import pandas as pd
import numpy as np

train = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/train.csv')
test = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/test.csv')

def analyze(df, name):
    print(f"\n===== Analyzing {name} =====")
    
    # Check for identical columns
    identical = []
    for i in range(len(df.columns)):
        for j in range(i + 1, len(df.columns)):
            if (df.iloc[:, i] == df.iloc[:, j]).all():
                identical.append((df.columns[i], df.columns[j]))
    
    if identical:
        print("Identical columns found:")
        for c1, c2 in identical:
            print(f"  - {c1} == {c2}")
    else:
        print("No identical columns found.")

    # Check Age sums
    age_parts = ['AGE_U18_PCT', 'AGE_18_24_PCT', 'AGE_25_34_PCT', 'AGE_35_44_PCT', 
                 'AGE_45_54_PCT', 'AGE_55_59_PCT', 'AGE_60_61_PCT', 'AGE_62_64_PCT', 
                 'AGE_65_69_PCT', 'AGE_70_79_PCT', 'AGE_80_PLUS_PCT']
    age_sum = df[age_parts].sum(axis=1)
    print(f"Sum of all Age parts (min/max): {age_sum.min():.4f} / {age_sum.max():.4f}")
    
    # Check Race sums
    race_parts = ['RACE_WHITE_NH_PCT', 'RACE_BLACK_NH_PCT', 'RACE_NATIVE_NH_PCT', 
                  'RACE_ASIAN_NH_PCT', 'RACE_PACIFIC_NH_PCT', 'RACE_TWO_OR_MORE_NH_PCT', 
                  'RACE_HISPANIC_ANY_PCT']
    race_sum = df[race_parts].sum(axis=1)
    print(f"Sum of all Race parts (min/max): {race_sum.min():.4f} / {race_sum.max():.4f}")

    # Out of range (PCT > 1.0)
    pct_cols = [c for c in df.columns if c.endswith('_PCT')]
    for col in pct_cols:
        if (df[col] > 1.0).any():
            count = (df[col] > 1.0).sum()
            print(f"COLUMN {col} has {count} values > 1.0! Max: {df[col].max():.4f}")

analyze(train, "Train")
analyze(test, "Test")

# Correlation with Target in Train
print("\n--- Correlation with HOMELESS_RATE ---")
numeric_train = train.select_dtypes(include=[np.number])
corrs = numeric_train.corr()['HOMELESS_RATE'].abs().sort_values(ascending=False)
print(corrs.head(15))

# Check for leakage: features in Test that have very different distribution or specific values
print("\n--- Feature Leakage / Shift Detection ---")
for col in test.columns:
    if col == 'ID': continue
    t_mean, te_mean = train[col].mean(), test[col].mean()
    t_std, te_std = train[col].std(), test[col].std()
    ratio = te_mean / t_mean if t_mean != 0 else 1
    if ratio > 1.5 or ratio < 0.5:
        print(f"Feature {col} mean shifted: Train={t_mean:.4f}, Test={te_mean:.4f} (Ratio={ratio:.2f})")

# Look for specific IDs that might be in both if they have different case or something
train_ids = set(train['ID'])
test_ids = set(test['ID'])
if train_ids.intersection(test_ids):
    print(f"ID Intersection: {train_ids.intersection(test_ids)}")

print("\n--- Summary of Mistakes ---")
print("1. Column 'AGE_U18_PCT' and 'FAMILY_MEMBERS_UNDER_18_PCT' are redundant (identical).")
print("2. Column 'NONFAMILY_SINGLE_FEMALE_PCT' and 'INDIVIDUALS_NOT_IN_FAMILY_UNITS_PCT' are redundant (identical).")
print("3. In Test, 'AGE_25_PLUS_PCT' has a value > 1.0 (1.3782).")
print("4. Age group percentages are inconsistent - they don't sum to 1.0 and their relative sums don't match 'AGE_25_PLUS_PCT'.")



from sklearn.metrics.pairwise import cosine_similarity
# Check for near duplicates using cosine similarity
sim = cosine_similarity(test.drop('ID',axis = 1), train.drop(['ID','HOMELESS_RATE'], axis = 1))
max_sim = sim.max(axis=1)
print(f"Max similarity between any row in Test and any row in Train: {max_sim.max():.6f}")

# Find rows with similarity > 0.9999
import numpy as np
indices = np.where(sim > 0.9999)
if len(indices[0]) > 0:
    print(f"Found {len(indices[0])} near duplicates!")
    for i, j in zip(indices[0], indices[1]):
        print(f"Test row {i} matches Train row {j} with similarity {sim[i,j]:.6f}")
else:
   print("No near duplicates found.")



import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def preprocess(df):
    # Redundant columns
    drop_cols = ['FAMILY_MEMBERS_UNDER_18_PCT', 'INDIVIDUALS_NOT_IN_FAMILY_UNITS_PCT']
    # If columns exist, drop them
    df = df.drop([c for c in drop_cols if c in df.columns], axis=1)
    
    # Clip AGE_25_PLUS_PCT if it's over 1.0 (Mistake in observation)
    if 'AGE_25_PLUS_PCT' in df.columns:
        df['AGE_25_PLUS_PCT'] = df['AGE_25_PLUS_PCT'].clip(upper=1.0)
    
    # Feature Engineering: ID Prefix
    df['Prefix'] = df['ID'].apply(lambda x: x.split('_')[0])
    df['Prefix'] = df['Prefix'].astype('category')
    
    return df

# Load data
# train = pd.read_csv('train.csv')
# test = pd.read_csv('test.csv')

# Preprocess
train_proc = preprocess(train)
test_proc = preprocess(test)

X = train_proc.drop(['ID', 'HOMELESS_RATE'], axis=1)
y = train_proc['HOMELESS_RATE']
X_test = test_proc.drop(['ID'], axis=1)

# Training with Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(test_proc))
mses = []

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'random_state': 42,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5
}

for i, (train_index, val_index) in enumerate(kf.split(X)):
    X_tr, X_val = X.iloc[train_index], X.iloc[val_index]
    y_tr, y_val = y.iloc[train_index], y.iloc[val_index]
    
    model = lgb.LGBMRegressor(**params,predict_disable_shape_check=True)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], 
              callbacks=[lgb.early_stopping(stopping_rounds=50)])
    
    preds_val = model.predict(X_val)
    mse = mean_squared_error(y_val, preds_val)
    mses.append(mse)
    print(f"Fold {i+1} RMSE: {np.sqrt(mse):.6f}")
    
    test_preds += model.predict(X_test) / kf.n_splits

print(f"\nAverage RMSE: {np.sqrt(np.mean(mses)):.6f}")

# Save submission
submission = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/sample_submission.csv')
submission['HOMELESS_RATE'] = test_preds
submission.to_csv('submission.csv', index=False)
print("\nsubmission.csv has been saved.")

# Feature importance
importances = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
importances = importances.sort_values(ascending=False, by='importance')
print("\nTop 10 Important Features:")
print(importances.head(10))


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def preprocess(df):
    # Redundant columns
    drop_cols = ['FAMILY_MEMBERS_UNDER_18_PCT', 'INDIVIDUALS_NOT_IN_FAMILY_UNITS_PCT']
    # If columns exist, drop them
    df = df.drop([c for c in drop_cols if c in df.columns], axis=1)
    
    # Clip AGE_25_PLUS_PCT if it's over 1.0 (Mistake in observation)
    if 'AGE_25_PLUS_PCT' in df.columns:
        df['AGE_25_PLUS_PCT'] = df['AGE_25_PLUS_PCT'].clip(upper=1.0)
    
    # Feature Engineering: ID Prefix
    df['Prefix'] = df['ID'].apply(lambda x: x.split('_')[0])
    df['Prefix'] = df['Prefix'].astype('category')
    
    return df

# Load data
train = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/train.csv')
test = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/test.csv')

# Preprocess
train_proc = preprocess(train)
test_proc = preprocess(test)

X = train_proc.drop(['ID', 'HOMELESS_RATE'], axis=1)
y = train_proc['HOMELESS_RATE']
X_test = test_proc.drop(['ID'], axis=1)

# Training with Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(test_proc))
mses = []

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'random_state': 42,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5
}

for i, (train_index, val_index) in enumerate(kf.split(X)):
    X_tr, X_val = X.iloc[train_index], X.iloc[val_index]
    y_tr, y_val = y.iloc[train_index], y.iloc[val_index]
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], 
              callbacks=[lgb.early_stopping(stopping_rounds=50)])
    
    preds_val = model.predict(X_val)
    mse = mean_squared_error(y_val, preds_val)
    mses.append(mse)
    print(f"Fold {i+1} RMSE: {np.sqrt(mse):.6f}")
    
    test_preds += model.predict(X_test) / kf.n_splits

print(f"\nAverage RMSE: {np.sqrt(np.mean(mses)):.6f}")

# Save submission
# submission = pd.read_csv('sample_submission.csv')
submission['HOMELESS_RATE'] = test_preds
submission.to_csv('submission.csv', index=False)
print("\nsubmission.csv has been saved.")

# Feature importance
importances = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
importances = importances.sort_values(ascending=False, by='importance')
print("\nTop 10 Important Features:")
print(importances.head(10))


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def preprocess(df):
    # 1. 冗長なカラムの削除 (内容が重複している列を排除)
    drop_redundant = ['FAMILY_MEMBERS_UNDER_18_PCT', 'INDIVIDUALS_NOT_IN_FAMILY_UNITS_PCT']
    df = df.drop([c for c in drop_redundant if c in df.columns], axis=1)
    
    # 2. データの誤りを修正 (パーセンテージを 0.0〜1.0 の範囲に制限)
    # これにより Test の異常値 (1.3782等) を 1.0 に補完します
    pct_cols = [col for col in df.columns if col.endswith('_PCT')]
    for col in pct_cols:
        df[col] = df[col].clip(0.0, 1.0)
    
    # 3. 地域情報を抽出 (Prefix: AL, LA, SF 等をカテゴリ特徴量化)
    df['Prefix'] = df['ID'].apply(lambda x: x.split('_')[0])
    df['Prefix'] = df['Prefix'].astype('category')
    
    return df

train_df = pd

# データの読み込みと前処理
train_proc = preprocess(train)
test_proc = preprocess(test)

X = train_proc.drop(['ID', 'HOMELESS_RATE'], axis=1)
y = train_proc['HOMELESS_RATE']
X_test = test_proc.drop(['ID'], axis=1)

# Cross-Validation (5-fold) による学習
kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(test_proc))
mses = []

params = {'objective': 'regression', 'metric': 'rmse', 'verbosity': -1, 'learning_rate': 0.05}

for i, (train_index, val_index) in enumerate(kf.split(X)):
    X_tr, X_val = X.iloc[train_index], X.iloc[val_index]
    y_tr, y_val = y.iloc[train_index], y.iloc[val_index]
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], 
              callbacks=[lgb.early_stopping(stopping_rounds=50)])
    
    # 予測の累積 (平均をとるため)
    test_preds += model.predict(X_test) / kf.n_splits
    mses.append(mean_squared_error(y_val, model.predict(X_val)))

print(f"Average RMSE: {np.sqrt(np.mean(mses)):.6f}")

# 提出用 csv の作成
submission = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/sample_submission.csv')
submission['HOMELESS_RATE'] = test_preds
submission.to_csv('submission.csv', index=False)

