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


# Load required libraries
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import gc


# Load main application data
app_train = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')
app_test = pd.read_csv('/kaggle/input/home-credit-default-risk/application_test.csv')

# Additional datasets
bureau = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau.csv')
bureau_balance = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau_balance.csv')
pos_cash = pd.read_csv('/kaggle/input/home-credit-default-risk/POS_CASH_balance.csv')
credit_card = pd.read_csv('/kaggle/input/home-credit-default-risk/credit_card_balance.csv')
installments = pd.read_csv('/kaggle/input/home-credit-default-risk/installments_payments.csv')


def aggregate_numeric(df, group_var, df_name):
    """Aggregates numeric variables by group_var."""
    numeric_df = df.select_dtypes(include=[np.number])
    agg = numeric_df.groupby(group_var).agg(['mean', 'sum', 'max', 'min'])
    agg.columns = [df_name + "_" + "_".join(col).strip() for col in agg.columns]
    agg.reset_index(inplace=True)
    return agg

def encode_categorical(df, group_var, df_name):
    """One-hot encodes categorical variables and aggregates them."""
    cat_df = pd.get_dummies(df.select_dtypes(include=['object']))
    cat_df[group_var] = df[group_var]
    agg = cat_df.groupby(group_var).sum()
    agg.columns = [df_name + "_" + col for col in agg.columns]
    return agg.reset_index()

### Process Bureau Data
bureau_agg = aggregate_numeric(bureau, 'SK_ID_CURR', 'bureau')
bureau_cat_agg = encode_categorical(bureau, 'SK_ID_CURR', 'bureau')
bureau_final = bureau_agg.merge(bureau_cat_agg, on='SK_ID_CURR', how='left')

### Process Bureau Balance Data
bureau_balance_agg = aggregate_numeric(bureau_balance, 'SK_ID_BUREAU', 'bureau_balance')
bureau_balance_cat_agg = encode_categorical(bureau_balance, 'SK_ID_BUREAU', 'bureau_balance')
bureau_balance_final = bureau_balance_agg.merge(bureau_balance_cat_agg, on='SK_ID_BUREAU', how='left')

# Merge with Bureau Data
bureau = bureau.merge(bureau_balance_final, on='SK_ID_BUREAU', how='left')
bureau.drop(columns=['SK_ID_BUREAU'], inplace=True)
bureau_final = aggregate_numeric(bureau, 'SK_ID_CURR', 'bureau_final')

### Process POS_CASH Data
pos_agg = aggregate_numeric(pos_cash, 'SK_ID_CURR', 'pos_cash')
pos_cat_agg = encode_categorical(pos_cash, 'SK_ID_CURR', 'pos_cash')
pos_final = pos_agg.merge(pos_cat_agg, on='SK_ID_CURR', how='left')

### Process Credit Card Data
credit_agg = aggregate_numeric(credit_card, 'SK_ID_CURR', 'credit_card')
credit_cat_agg = encode_categorical(credit_card, 'SK_ID_CURR', 'credit_card')
credit_final = credit_agg.merge(credit_cat_agg, on='SK_ID_CURR', how='left')

### Process Installment Payments Data
install_agg = aggregate_numeric(installments, 'SK_ID_CURR', 'installments')

### Merge All Features into Application Data
app_train = app_train.merge(bureau_final, on='SK_ID_CURR', how='left')
app_test = app_test.merge(bureau_final, on='SK_ID_CURR', how='left')

app_train = app_train.merge(pos_final, on='SK_ID_CURR', how='left')
app_test = app_test.merge(pos_final, on='SK_ID_CURR', how='left')

app_train = app_train.merge(credit_final, on='SK_ID_CURR', how='left')
app_test = app_test.merge(credit_final, on='SK_ID_CURR', how='left')

app_train = app_train.merge(install_agg, on='SK_ID_CURR', how='left')
app_test = app_test.merge(install_agg, on='SK_ID_CURR', how='left')

# Fill missing values
app_train.fillna(0, inplace=True)
app_test.fillna(0, inplace=True)

print("Final training shape:", app_train.shape)
print("Final test shape:", app_test.shape)


# Prepare training data
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import numpy as np

# Separate features and target
X = app_train.drop(columns=['TARGET', 'SK_ID_CURR'])
y = app_train['TARGET']
test_X = app_test.drop(columns=['SK_ID_CURR'])

# Ensure data types are numeric
X = X.select_dtypes(include=[np.number])
test_X = test_X.select_dtypes(include=[np.number])

# Split into train and validation sets (80-20 split)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training data shape:", X_train.shape)
print("Validation data shape:", X_valid.shape)
print("Test data shape:", test_X.shape)


# Train XGBoost model
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# Initialize XGBoost model with early stopping
xgb_model = XGBClassifier(
    n_estimators=500,  
    learning_rate=0.05,  
    max_depth=6,  
    subsample=0.8,  
    colsample_bytree=0.8,  
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    early_stopping_rounds=50,  # Move early stopping here
    random_state=42
)

# Train model
xgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=100)

# Validate model performance
valid_preds = xgb_model.predict_proba(X_valid)[:, 1]
auc_score = roc_auc_score(y_valid, valid_preds)
print(f"Validation AUC: {auc_score:.4f}")


# Make submission file
# Make predictions on test data
test_preds = xgb_model.predict_proba(test_X)[:, 1]

# Create submission DataFrame
submission = pd.DataFrame({'SK_ID_CURR': app_test['SK_ID_CURR'], 'TARGET': test_preds})

# Save to CSV
submission.to_csv('submission.csv', index=False)

print("Submission file saved!")

