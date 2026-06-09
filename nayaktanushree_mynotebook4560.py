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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import numpy as np


data_dir = '/kaggle/input/playground-series-s5e11/'
train_df = pd.read_csv(data_dir + 'train.csv')
test_df = pd.read_csv(data_dir + 'test.csv')
submission_df = pd.read_csv(data_dir + 'sample_submission.csv')

TARGET = 'loan_paid_back'
ID = 'id'


numeric_cols = train_df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = train_df.select_dtypes(include='object').columns.tolist()

if ID in numeric_cols:
    numeric_cols.remove(ID)
if TARGET in numeric_cols:
    numeric_cols.remove(TARGET)

print(f"Numerical Features ({len(numeric_cols)}): {numeric_cols[:3]}...")
print(f"Categorical Features ({len(categorical_cols)}): {categorical_cols}")


imputer = SimpleImputer(strategy='median')
imputer.fit(train_df[numeric_cols])

train_df[numeric_cols] = imputer.transform(train_df[numeric_cols])
test_df[numeric_cols] = imputer.transform(test_df[numeric_cols])


for col in categorical_cols:
    combined_data = pd.concat([train_df[col], test_df[col]], axis=0).astype(str).fillna('missing')
    le = LabelEncoder()
    le.fit(combined_data)
    train_df[col] = le.transform(train_df[col].astype(str).fillna('missing'))
    test_df[col] = le.transform(test_df[col].astype(str).fillna('missing'))


features = [col for col in train_df.columns if col not in [ID, TARGET]]

X_train = train_df[features]
y_train = train_df[TARGET]

X_test = test_df[features]

print(f"Training features shape: {X_train.shape}")
print(f"Training target shape: {y_train.shape}")
print(f"Test features shape: {X_test.shape}")


train_df['debt_to_income_ratio'] = train_df['loan_amount'] / (train_df['annual_income'] + 1e-6)
test_df['debt_to_income_ratio'] = test_df['loan_amount'] / (test_df['annual_income'] + 1e-6)

train_df['loan_score_ratio'] = train_df['loan_amount'] / (train_df['credit_score'] + 1e-6)
test_df['loan_score_ratio'] = test_df['loan_amount'] / (test_df['credit_score'] + 1e-6)

print("New ratio features created.")


features = [col for col in train_df.columns if col not in [ID, TARGET]]


test_ids = test_df['id']
print("Test IDs saved successfully.")


del train_df, test_df, submission_df
import gc
gc.collect()


lgb_clf = lgb.LGBMClassifier(
    objective='binary',          
    metric='auc',                
    n_estimators=1000,           
    learning_rate=0.05,          
    random_state=42,             
    n_jobs=-1,                   
    verbose=-1                   
)


print("Starting LightGBM Model Training...")

lgb_clf.fit(X_train, y_train)

print("Training Complete!")


print("Generating predictions on the test set...")

y_pred_proba = lgb_clf.predict_proba(X_test)

predictions = y_pred_proba[:, 1]

print("Predictions complete. A sample of the first 5 probabilities:")
print(predictions[:5])


lgbm_only_submission = pd.DataFrame({
    'id': test_ids,
    'loan_paid_back': predictions
})

lgbm_only_submission.to_csv('lgbm_only_submission.csv', index=False)
print("LGBM submission file 'lgbm_only_submission.csv' is ready for submission and scoring.")


import catboost as cb
from catboost import CatBoostClassifier


cat_clf = CatBoostClassifier(
    iterations=500,
    learning_rate=0.08,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    verbose=0 
)

cat_clf.fit(X_train, y_train) 
print("CatBoost Training Complete!")


print("Generating predictions with CatBoost...")
cat_predictions = cat_clf.predict_proba(X_test)[:, 1] 
print("CatBoost predictions complete.")


catboost_only_submission = pd.DataFrame({
    'id': test_ids,
    'loan_paid_back': cat_predictions
})

catboost_only_submission.to_csv('catboost_only_submission.csv', index=False)
print("CatBoost submission file 'catboost_only_submission.csv' is ready for submission and scoring.")


print("Generating predictions with CatBoost...")

cat_predictions = cat_clf.predict_proba(X_test)[:, 1]

print("CatBoost predictions complete.")


ensemble_predictions = (predictions + cat_predictions) / 2

print("Ensembling complete. Combined predictions sample:")
print(ensemble_predictions[:5])


ensemble_predictions = (cat_predictions * 0.6) + (predictions * 0.4) 

print("Ensembling complete. Using a 60/40 weighted average.")


ensemble_predictions = (predictions + cat_predictions) / 2 

ensemble_submission = pd.DataFrame({
    'id': test_ids,
    'loan_paid_back': ensemble_predictions
})

ensemble_submission.to_csv('ensemble_submission.csv', index=False) 

print("Ensemble submission file 'ensemble_submission.csv' successfully created!")


ensemble_submission.to_csv('ensemble_submission.csv', index=False)
print("Final file saved.")

