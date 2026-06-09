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


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

# Load Data
df_train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')



# Identify target variable
TARGET_COL = "Listening_Time_minutes"  # Adjust if needed

# Drop Unnecessary Columns
drop_cols = ["id", "Podcast_Name"]  # Adjust based on dataset
df_train.drop(columns=drop_cols, inplace=True, errors='ignore')
df_test.drop(columns=drop_cols, inplace=True, errors='ignore')

# Identify categorical and numerical columns
categorical_cols = df_train.select_dtypes(include=['object']).columns
numerical_cols = df_train.select_dtypes(include=['number']).columns.drop(TARGET_COL, errors='ignore')

# Handle Missing Values
imputer_num = SimpleImputer(strategy='median')
imputer_cat = SimpleImputer(strategy='most_frequent')

df_train[numerical_cols] = imputer_num.fit_transform(df_train[numerical_cols])
df_test[numerical_cols] = imputer_num.transform(df_test[numerical_cols])

df_train[categorical_cols] = imputer_cat.fit_transform(df_train[categorical_cols])
df_test[categorical_cols] = imputer_cat.transform(df_test[categorical_cols])

# Encode Categorical Variables
for col in categorical_cols:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col])
    df_test[col] = le.transform(df_test[col])

# Feature Scaling
scaler = StandardScaler()
df_train[numerical_cols] = scaler.fit_transform(df_train[numerical_cols])
df_test[numerical_cols] = scaler.transform(df_test[numerical_cols])

# Define Features & Target
X = df_train.drop(columns=[TARGET_COL])
y = df_train[TARGET_COL]

# Train/Test Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert Data to LightGBM Format
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val)


params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'verbose': -1
}

# Train Model with Early Stopping
model = lgb.train(
    params,
    train_data,
    valid_sets=[val_data],  # Validation set
    num_boost_round=10000,
    callbacks=[  
        lgb.early_stopping(50),  # ✅ Early stopping
        lgb.log_evaluation(100)  # ✅ Replaces `verbose` or `verbose_eval`
    ]
)




# Predictions & Evaluation
y_pred = model.predict(X_val, num_iteration=model.best_iteration)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE: {rmse:.4f}")

# Predictions on Test Set
test_preds = model.predict(df_test, num_iteration=model.best_iteration)

# Save Submission
submission = pd.DataFrame({'id': df_test.index, TARGET_COL: test_preds})
submission.to_csv('submission.csv', index=False)


df_test['id'] = range(1, len(df_test) + 1)



print(df_test.columns)  # Check if there are duplicate column names
print(df_test[['id']].head())  # See what's inside 'id'
print(df_test['id'].values.shape)  # Should be (250000,) if it's correct



submission = pd.DataFrame({
    'id': df_test['id'],
    'Listening_Time_minutes': test_preds
})

submission.to_csv('submission.csv', index=False)
print("✅ Submission file saved successfully!")














