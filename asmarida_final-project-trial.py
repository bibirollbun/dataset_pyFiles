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


# Load packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from IPython.display import display
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.metrics import roc_auc_score



# Load datasets
train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')
train_identity = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')
test_transaction = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')
test_identity = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv')

# Basic info about the files
print("Train Transaction shape:", train_transaction.shape)
print("Train Identity shape:", train_identity.shape)
print("Test Transaction shape:", test_transaction.shape)
print("Test Identity shape:", test_identity.shape)

# Preview the data
print(train_transaction.head())
print(train_identity.head())

# See all the columns
print(train_transaction.columns)
print(train_identity.columns)


# Check missing values
print(train_transaction.isnull().mean().sort_values(ascending=False))
print(train_identity.isnull().mean().sort_values(ascending=False))

# Check constant columns (same value everywhere)
constant_cols_trans = [col for col in train_transaction.columns if train_transaction[col].nunique() <= 1]
constant_cols_id    = [col for col in train_identity.columns if train_identity[col].nunique() <= 1]
print("Constant columns in transaction:", constant_cols_trans)
print("Constant columns in identity:", constant_cols_id)


# combine transaction with the identity 
train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')


print(f'Train dataset has {train.shape[0]} rows and {train.shape[1]} columns.')
print(f'Test dataset has {test.shape[0]} rows and {test.shape[1]} columns.')


train_transaction.head()


train_identity.head()


#del train_identity, train_transaction, test_identity, test_transaction


one_value_cols = [col for col in train.columns if train[col].nunique() <= 1]
one_value_cols_test = [col for col in test.columns if test[col].nunique() <= 1]
one_value_cols == one_value_cols_test


print(f'There are {len(one_value_cols)} columns in train dataset with one unique value.')
print(f'There are {len(one_value_cols_test)} columns in test dataset with one unique value.')


print(one_value_cols_test)


print(f'There are {train.isnull().any().sum()} columns in train dataset with missing values.')


print(train.isnull().mean().sort_values(ascending=False))

# Check target balance
print(train['isFraud'].value_counts(normalize=True))


import matplotlib.pyplot as plt
import seaborn as sns

# Count plot
plt.figure(figsize=(6,4))
sns.countplot(x='isFraud', data=train)
plt.title("Fraud vs Non-Fraud Count", fontsize=16)
plt.show()

# Percentage
fraud_rate = train['isFraud'].mean() * 100
print(f"Fraud Rate: {fraud_rate:.2f}%")


# Step 1: Show missing values
missing_values = train.isnull().mean().sort_values(ascending=False)

# Step 2: Plot top 30 missing
missing_values.head(30).plot(kind='barh', figsize=(12,8))
plt.title("Top 30 Columns with Most Missing Values", fontsize=16)
plt.show()

# Step 3: List columns with >90% missing
high_missing_cols = train.columns[train.isnull().mean() > 0.90]
print("Columns with >90% missing:", list(high_missing_cols))

# Step 4: Drop those columns
train.drop(columns=high_missing_cols, inplace=True)
test.drop(columns=high_missing_cols, inplace=True, errors='ignore')  # <-- safer with errors='ignore'

print(f"âœ… New train shape: {train.shape}")
print(f"âœ… New test shape: {test.shape}")


charts = {}
for i in ['ProductCD', 'card4', 'card6', 'M4', 'M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9']:
    feature_count = train[i].value_counts(dropna=False).reset_index()
    feature_count.columns = [i, 'count']  # Correctly name the columns
    chart = alt.Chart(feature_count).mark_bar().encode(
                y=alt.Y(f"{i}:N", axis=alt.Axis(title=i)),
                x=alt.X('count:Q', axis=alt.Axis(title='Count')),
                tooltip=[i, 'count']
            ).properties(title=f"Counts of {i}", width=400)
    charts[i] = chart

display(
    (charts['ProductCD'] | charts['card4']) &
    (charts['card6'] | charts['M4']) &
    (charts['M1'] | charts['M2']) &
    (charts['M3'] | charts['M5']) &
    (charts['M6'] | charts['M7']) &
    (charts['M8'] | charts['M9'])
)


# Features to plot
features = ['ProductCD', 'card4', 'DeviceType', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']

# Set up the grid: 3 columns
n_cols = 3
n_rows = (len(features) + n_cols - 1) // n_cols  # automatically calculate number of rows needed

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 5))  # adjust figsize based on rows

# Flatten axes for easy indexing
axes = axes.flatten()

# Loop through features
for i, feature in enumerate(features):
    sns.countplot(x=feature, hue='isFraud', data=train, ax=axes[i])
    axes[i].set_title(f"{feature} vs Fraud", fontsize=14)
    axes[i].tick_params(axis='x', rotation=45)  # rotate if needed

# Turn off empty subplots if any
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


for feature in ['P_emaildomain', 'R_emaildomain', 'card1', 'card2', 'card3', 'card5', 'addr1', 'addr2']:
    print(f"Top values for {feature}:")
    print(train[feature].value_counts(dropna=False).head(10))
    print('-'*50)


charts = {}

for i in ['P_emaildomain', 'R_emaildomain', 'card1', 'card2', 'card3', 'card5', 'addr1', 'addr2']:
    feature_count = train[i].value_counts(dropna=False).reset_index()[:40]
    feature_count.columns = [i, 'count']  # properly name columns
    chart = alt.Chart(feature_count).mark_bar().encode(
        x=alt.X(f"{i}:N", axis=alt.Axis(title=i)),
        y=alt.Y('count:Q', axis=alt.Axis(title='Count')),
        tooltip=[i, 'count']
    ).properties(
        title=f"Counts of {i}",
        width=600
    )
    charts[i] = chart
    
display(
    (charts['P_emaildomain'] | charts['R_emaildomain']) &
    (charts['card1'] | charts['card2']) &
    (charts['card3'] | charts['card5']) &
    (charts['addr1'] | charts['addr2'])
)


# Add log-transformed amount safely
train['TransactionAmt_log'] = np.log1p(train['TransactionAmt']) if 'TransactionAmt_log' not in train.columns else train['TransactionAmt_log']

# Features you want to plot
num_features = ['TransactionAmt', 'TransactionAmt_log', 'dist1', 'dist2']

# Set grid size dynamically
n_cols = 2
n_rows = (len(num_features) + n_cols - 1) // n_cols  # auto-calculate number of rows

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 5))
axes = axes.flatten()

# Plot each feature
for idx, feature in enumerate(num_features):
    if feature in train.columns:
        sns.histplot(train[feature], bins=100, kde=True, ax=axes[idx])
        axes[idx].set_title(f"{feature} Distribution", fontsize=16)
        axes[idx].tick_params(axis='x', rotation=45)
    else:
        axes[idx].set_visible(False)

# Tidy layout
plt.tight_layout()
plt.show()


# Correlation with target (only numerics)
numerical_cols = train.select_dtypes(include=[np.number]).columns.tolist()
numerical_cols.remove('isFraud')

corr = train[numerical_cols + ['isFraud']].corr()['isFraud'].sort_values(ascending=False)

print("Top correlated features with isFraud:")
print(corr.head(20))  # top positive
print(corr.tail(20))  # top negative

# Correlation heatmap
plt.figure(figsize=(16,12))
sns.heatmap(train[numerical_cols].corr(), cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Feature Correlation Heatmap", fontsize=16)
plt.show()


top_features = corr.index[:20].tolist() + corr.index[-20:].tolist()  # top and bottom 20
top_features = list(set(top_features))  # remove duplicates

plt.figure(figsize=(14,10))
sns.heatmap(train[top_features].corr(), cmap='coolwarm', vmin=-1, vmax=1, annot=False)
plt.title("Top 20 Positive/Negative Correlated Features Heatmap", fontsize=16)
plt.show()


plt.hist(train['TransactionDT'], label='train')
plt.hist(test['TransactionDT'], label='test')
plt.legend()
plt.title('Distribution of transaction dates')



# Check columns with more than 90% missing values
missing_percent = train.isnull().mean() * 100  # percent missing
high_missing_cols = missing_percent[missing_percent > 90]

# Display them
if high_missing_cols.empty:
    print("No columns with >90% missing values remain.")
else:
    print("Columns with >90% missing values still present:")
    print(high_missing_cols)


# Drop from train
#train.drop(columns=high_missing_cols, inplace=True, errors='ignore')

# Drop from test 
#test.drop(columns=high_missing_cols, inplace=True, errors='ignore')


# Check how many columns are remaining in train and test
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Recalculate missing values after drop
missing_after_drop = train.isnull().mean().sort_values(ascending=False)

# Print top 20 columns with missing values remaining
print("Top 20 columns still having missing values (after dropping high-missing ones):")
print(missing_after_drop.head(20))

# Check if any column still has >90% missing
high_missing_after_drop = missing_after_drop[missing_after_drop > 0.90]

print(f"\nColumns still with >90% missing after drop: {list(high_missing_after_drop.index)}")


# Fill missing values for numeric columns
numeric_cols = train.select_dtypes(include=[np.number]).columns

# Only use numeric columns that exist in both train and test
numeric_cols_existing = [col for col in numeric_cols if col in test.columns]

train[numeric_cols] = train[numeric_cols].fillna(-1)
test[numeric_cols_existing] = test[numeric_cols_existing].fillna(-1)

# Fill missing values for object (categorical) columns
categorical_cols = train.select_dtypes(include=['object']).columns
categorical_cols_existing = [col for col in categorical_cols if col in test.columns]

train[categorical_cols] = train[categorical_cols].fillna('missing')
test[categorical_cols_existing] = test[categorical_cols_existing].fillna('missing')


print("âœ… No missing values left in train!") if train.isnull().sum().sum() == 0 else print("âš ï¸� Still missing values in train.")

print("âœ… No missing values left in test!") if test.isnull().sum().sum() == 0 else print("âš ï¸� Still missing values in test.")


# Find columns with missing values in test
missing_in_test = test.columns[test.isnull().any()].tolist()
print("Columns still with missing in test:", missing_in_test)



# Fill missing numeric columns with -1
for col in missing_in_test:
    if test[col].dtype in ['float64', 'int64']:
        test[col] = test[col].fillna(-1)
    else:
        test[col] = test[col].fillna('missing')


print("âœ… No missing values left in test!") if test.isnull().sum().sum() == 0 else print("âš ï¸� Still missing values in test.")


# Find categorical columns in train
categorical_cols = train.select_dtypes(include=['object']).columns

# Loop through each categorical column
for col in categorical_cols:
    le = LabelEncoder()
    
    if col in test.columns:  # Only if column exists in both train and test
        combined_data = pd.concat([train[col], test[col]], axis=0).astype(str)
        le.fit(combined_data)
        
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
        
    else:  # If column only exists in train, just encode train
        train[col] = le.fit_transform(train[col].astype(str))


for feature in ['id_03', 'id_04', 'id_30', 'id_31', 'id_32', 'id_09', 'id_10']:
    print(f"{feature}: {train[feature].nunique()} unique values")


# Create 'TransactionDT_days' to represent how many days passed
train['TransactionDT_days'] = train['TransactionDT'] / (24 * 60 * 60)
test['TransactionDT_days'] = test['TransactionDT'] / (24 * 60 * 60)

# Create 'hour' feature
train['Transaction_hour'] = ((train['TransactionDT'] / 3600) % 24).astype(int)
test['Transaction_hour'] = ((test['TransactionDT'] / 3600) % 24).astype(int)

# Create 'weekday' feature
train['Transaction_weekday'] = ((train['TransactionDT'] / (3600*24)) % 7).astype(int)
test['Transaction_weekday'] = ((test['TransactionDT'] / (3600*24)) % 7).astype(int)

# Create 'is_weekend' feature
train['is_weekend'] = (train['Transaction_weekday'] >=5).astype(int)
test['is_weekend'] = (test['Transaction_weekday'] >=5).astype(int)



plt.figure(figsize=(10,6))
train['Transaction_hour'].hist(bins=24)
plt.title('Transaction Hour Distribution (Train)', fontsize=16)
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Transactions')
plt.show()


plt.figure(figsize=(10,6))
train['Transaction_weekday'].hist(bins=7)
plt.title('Transaction Weekday Distribution (Train)', fontsize=16)
plt.xlabel('Day of the Week')
plt.ylabel('Number of Transactions')
plt.show()


plt.figure(figsize=(6,4))
train['is_weekend'].value_counts().plot(kind='bar')
plt.title('Weekend vs Weekday Transactions (Train)', fontsize=16)
plt.xlabel('Is Weekend')
plt.ylabel('Number of Transactions')
plt.show()


# See which categories will be grouped for each feature
for col in ['DeviceType', 'card4']:
    freq = train[col].value_counts()
    rare_cats = freq[freq < 1000].index
    print(f"Feature: {col}")
    print(f"Categories to be replaced with 'Rare': {list(rare_cats)}")
    print('-'*50)


# Find features with too many unique values
for col in ['P_emaildomain', 'R_emaildomain', 'id_30', 'id_31', 'id_33', 'id_34', 'card1', 'card2', 'card5']:
    print(f"{col}: {train[col].nunique()} unique values")


card1_freq = train['card1'].value_counts()
train['card1_freq'] = train['card1'].map(card1_freq)
test['card1_freq'] = test['card1'].map(card1_freq)


# Group rare categories 
# Features to group rare categories
grouping_features = ['P_emaildomain', 'R_emaildomain', 'id_30', 'id_31', 'id_33', 'card2', 'card5']

for col in grouping_features:
    freq = train[col].value_counts()
    rare_cats = freq[freq < 500].index  # you can adjust 500 based on how strict you want
    train[col] = train[col].replace(rare_cats, 'Rare')
    if col in test.columns:
        test[col] = test[col].replace(rare_cats, 'Rare')
print("Rare categories grouped.")

# Create transaction amount ratios 
for col in ['card1', 'card4']:
    train[f'TransactionAmt_to_mean_{col}'] = train['TransactionAmt'] / train.groupby(col)['TransactionAmt'].transform('mean')
    test[f'TransactionAmt_to_mean_{col}'] = test['TransactionAmt'] / test.groupby(col)['TransactionAmt'].transform('mean')

    train[f'TransactionAmt_to_std_{col}'] = train['TransactionAmt'] / train.groupby(col)['TransactionAmt'].transform('std')
    test[f'TransactionAmt_to_std_{col}'] = test['TransactionAmt'] / test.groupby(col)['TransactionAmt'].transform('std')

print("Transaction amount ratio features created.")

# Create "is_nighttime" Feature 
train['is_nighttime'] = ((train['Transaction_hour'] >= 0) & (train['Transaction_hour'] <= 5)).astype(int)
test['is_nighttime'] = ((test['Transaction_hour'] >= 0) & (test['Transaction_hour'] <= 5)).astype(int)

print("Nighttime feature created.")

# Preserve transactionID for later
test_transaction_id = test['TransactionID'].copy()

# Drop TransactionID 
drop_cols = ['TransactionID']
train.drop(columns=[col for col in drop_cols if col in train.columns], inplace=True)
test.drop(columns=[col for col in drop_cols if col in test.columns], inplace=True)

print("Dropped useless columns.")

# Final Check 
print(f"Train shape after feature engineering: {train.shape}")
print(f"Test shape after feature engineering: {test.shape}")


print("Columns in test but not in train:", set(test.columns) - set(train.columns))
print("Columns in train but not in test:", set(train.columns) - set(test.columns))


# Fix column names in test
test.columns = test.columns.str.replace('-', '_')


print("Columns in test but not in train:", set(test.columns) - set(train.columns))
print("Columns in train but not in test:", set(train.columns) - set(test.columns))


test[['id_07', 'id_08', 'id_18', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27']].describe()


# See missing percentages
missing_test = test[['id_07', 'id_08', 'id_18', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27']].isnull().mean() * 100
print(missing_test.sort_values())


# Drop extra ID columns from test
extra_cols_in_test = ['id_07', 'id_08', 'id_18', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27']
test = test.drop(columns=extra_cols_in_test)

# Create TransactionAmt_log for test set
test['TransactionAmt_log'] = np.log1p(test['TransactionAmt'])


print("Columns in test but not in train:", set(test.columns) - set(train.columns))
print("Columns in train but not in test:", set(train.columns) - set(test.columns))


#train = train.copy()
#test = test.copy()


cols_high_missing = ['id_30', 'id_31', 'id_33', 'id_34', 'id_28', 'id_29']
# Check missing values percentage for high-missing columns
missing_percent = train[cols_high_missing].isnull().mean() * 100
print(missing_percent.sort_values())

# Also, basic description (to see weird values)
print(train[cols_high_missing].describe(include='all'))


# Drop only id_34 (because it's useless)
train.drop(columns=['id_34'], inplace=True)
test.drop(columns=['id_34'], inplace=True)


# Columns you are checking
low_missing_cols = ['id_12', 'id_15', 'id_16', 'id_35', 'id_36', 'id_37', 'id_38']

# Check data types
for col in low_missing_cols:
    if col in train.columns:
        print(f"{col} - dtype: {train[col].dtype}, unique values: {train[col].nunique()}")


from sklearn.preprocessing import LabelEncoder

# Columns that are causing problem
cat_cols = ['card2', 'card5', 'P_emaildomain', 'R_emaildomain', 'id_30', 'id_31', 'id_33']

# Label Encode
for col in cat_cols:
    if col in train.columns:
        le = LabelEncoder()
        all_data = pd.concat([train[col], test[col]], axis=0).astype(str)  # Fit on both train and test
        le.fit(all_data)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))

print("âœ… Categorical columns encoded successfully!")


train = train.copy()
test = test.copy()


# Define target and features
TARGET = 'isFraud'
features = [col for col in train.columns if col != TARGET]


# Sort by TransactionDT_days for proper time order
train = train.sort_values('TransactionDT_days').reset_index(drop=True)

# Define train/validation split (80% train, 20% validation)
split_index = int(len(train) * 0.8)

X_train = train.iloc[:split_index][features]
y_train = train.iloc[:split_index][TARGET]

X_val = train.iloc[split_index:][features]
y_val = train.iloc[split_index:][TARGET]

print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}")


import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np

# Define LightGBM parameters
lgb_params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'metric': 'auc',
    'n_jobs': -1,
    'learning_rate': 0.005,
    'num_leaves': 128,
    'max_depth': -1,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'scale_pos_weight': 20,
    'random_state': 2025,
    'verbose': -1
}

# Prepare X and y
X = train[features]
y = train[TARGET]

# Fix test set object columns before starting
for col in test.columns:
    if test[col].dtype == 'object':
        test[col] = test[col].astype('category').cat.codes

# Stratified K-Fold CV
n_splits = 3
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2025)

# Store out-of-fold predictions
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test))

# Train model on each fold
for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"\nğŸ“š Training fold {fold + 1}...")
    
    X_train_cv, X_valid_cv = X.iloc[train_idx], X.iloc[valid_idx]
    y_train_cv, y_valid_cv = y.iloc[train_idx], y.iloc[valid_idx]
    
    train_set = lgb.Dataset(X_train_cv, label=y_train_cv)
    valid_set = lgb.Dataset(X_valid_cv, label=y_valid_cv)
    
    model = lgb.train(
        lgb_params,
        train_set,
        num_boost_round=5000,
        valid_sets=[train_set, valid_set],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200),
            lgb.log_evaluation(period=100)
        ]
    )
    
    # Predictions
    oof_preds[valid_idx] = model.predict(X_valid_cv)
    test_preds += model.predict(test[features]) / n_splits

# Final validation AUC
final_auc = roc_auc_score(y, oof_preds)
print(f"\nâœ… Final Cross-Validated AUC: {final_auc:.5f}")

# Save test predictions
lgb_test_preds = test_preds


# Clean infinity values before DMatrix creation
x_train = x_train.replace([np.inf, -np.inf], np.nan)
x_valid = x_valid.replace([np.inf, -np.inf], np.nan)
x_test  = x_test.replace([np.inf, -np.inf], np.nan)

# Then fill any NaNs (missing) safely
x_train = x_train.fillna(-999)
x_valid = x_valid.fillna(-999)
x_test  = x_test.fillna(-999)


import xgboost as xgb
from sklearn.metrics import roc_auc_score

# Better tuned XGBoost parameters
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'n_jobs': -1,
    'learning_rate': 0.005,  # Slower, better convergence
    'max_depth': 7,          # Slightly deeper trees for complexity
    'subsample': 0.85,       # Randomness to avoid overfitting
    'colsample_bytree': 0.85,
    'scale_pos_weight': 20,  # Important for fraud rarity
    'min_child_weight': 5,   # Helps regularize leaf nodes
    'gamma': 0.1,            # Minimum loss reduction to make a split
    'lambda': 1.0,           # L2 regularization
    'alpha': 0.5,            # L1 regularization
    'random_state': 2025,
}

# Prepare DMatrix
xgb_train = xgb.DMatrix(X_train, label=y_train)
xgb_valid = xgb.DMatrix(X_val, label=y_val)
xgb_test = xgb.DMatrix(X_test)

# Train
print("Training XGBoost model...")
xgb_model = xgb.train(
    xgb_params,
    dtrain=xgb_train,
    num_boost_round=20000,   # Allow longer training
    evals=[(xgb_train, 'train'), (xgb_valid, 'valid')],
    early_stopping_rounds=200,
    verbose_eval=100
)

# Validation prediction
y_val_pred_xgb = xgb_model.predict(xgb_valid)
val_score_xgb = roc_auc_score(y_val, y_val_pred_xgb)

print(f"\nâœ… Validation ROC-AUC (XGBoost): {val_score_xgb:.5f}")

# Test prediction
xgb_test_preds = xgb_model.predict(xgb_test)


from xgboost import XGBClassifier
import joblib  # for saving model
from sklearn.metrics import roc_auc_score

# Better tuned XGBClassifier params
xgbc_model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    n_jobs=-1,
    learning_rate=0.005,
    max_depth=7,
    subsample=0.85,
    colsample_bytree=0.85,
    scale_pos_weight=20,
    min_child_weight=5,
    gamma=0.1,
    reg_lambda=1.0,
    reg_alpha=0.5,
    random_state=2025,
    n_estimators=20000,
    verbosity=1
)

# Train XGBClassifier
print("Training XGBClassifier model...")
xgbc_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=200,
    verbose=100
)

# Validation prediction
y_val_pred_xgbc = xgbc_model.predict_proba(X_val)[:, 1]
val_score_xgbc = roc_auc_score(y_val, y_val_pred_xgbc)
print(f"\nâœ… Validation ROC-AUC (XGBClassifier): {val_score_xgbc:.5f}")

# Test prediction
xgbc_test_preds = xgbc_model.predict_proba(X_test)[:, 1]


# Save LightGBM model
model.save_model('lgb_model.txt')


# Save LightGBM model
lgb.train_model.save_model('lgb_model.txt')

# Save XGBoost model
xgb_model.save_model('xgb_model.json')

# Save XGBClassifier model
joblib.dump(xgbc_model, 'xgbc_model.pkl')


# Save test set (no target label)
test_with_id = test.copy()
test_with_id['TransactionID'] = test_transaction_id  # Restore TransactionID
test_with_id.to_csv('final_test_set.csv', index=False)


submission_lgb = pd.DataFrame({
    'TransactionID': test_transaction_id,
    'isFraud': lgb_test_preds
})
submission_lgb.to_csv('submission_lgb.csv', index=False)



 submission_xgb = pd.DataFrame({
    'TransactionID': test_transaction_id,
    'isFraud': xgb_test_preds
})
submission_xgb.to_csv('submission_xgb.csv', index=False)


submission_xgbc = pd.DataFrame({
    'TransactionID': test_transaction_id,
    'isFraud': xgbc_test_preds
})
submission_xgbc.to_csv('submission_xgbc.csv', index=False)


# Simple average of all 3 models
blended_preds = (lgb_test_preds + xgb_test_preds + xgbc_test_preds) / 3

submission_blend = pd.DataFrame({
    'TransactionID': test_transaction_id,
    'isFraud': blended_preds
})
submission_blend.to_csv('submission_blend.csv', index=False)

