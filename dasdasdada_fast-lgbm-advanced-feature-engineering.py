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
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# 1. LOAD DATA
train_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")

# 2. ADVANCED FEATURE ENGINEERING
for df in [train_df, test_df]:
    # From the first round of feature ideas
    df['balance_per_age'] = df['balance'] / (df['age'] + 1)
    df['duration_x_campaign'] = df['duration'] * df['campaign']
    
    # From the second round of feature ideas
    df['was_previously_contacted'] = (df['pdays'] != -1).astype(int)
    df['age_group'] = pd.cut(df['age'], 
                             bins=[17, 30, 40, 50, 60, 100], 
                             labels=['Young', 'Young-Adult', 'Adult', 'Senior', 'Elderly'])

# 3. PREPARE DATA FOR MODELING
X = train_df.drop(["id", "y"], axis=1)
y = train_df["y"]
X_test = test_df.drop("id", axis=1)

# Convert the new 'age_group' column to 'object' type for the pipeline
X['age_group'] = X['age_group'].astype('object')
X_test['age_group'] = X_test['age_group'].astype('object')

# 4. DEFINE PREPROCESSING PIPELINE
# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['number']).columns

# Create the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# 5. DEFINE THE MODEL (with strong, pre-set parameters)
# These are solid parameters that serve as a great starting point without tuning
strong_params = {
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'n_estimators': 1500,
    'learning_rate': 0.02,
    'num_leaves': 40,
    'max_depth': 8,
    'lambda_l1': 0.5,
    'lambda_l2': 0.5,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'random_state': 42
}

# 6. CREATE AND TRAIN THE FINAL PIPELINE
print("Training the final model...")
final_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', lgb.LGBMClassifier(**strong_params))])

final_pipeline.fit(X, y)
print("Model training complete!")

# 7. CREATE SUBMISSION FILE
print("Creating submission file...")
test_predictions = final_pipeline.predict_proba(X_test)[:, 1]

submission_df = pd.DataFrame({'id': test_df['id'], 'y': test_predictions})
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' created successfully!")
print(submission_df.head())

