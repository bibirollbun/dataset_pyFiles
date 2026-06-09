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


#Importing libraries..
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt


df_train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
df_submission = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


df_train.info()


#df_test.info()


#df_submission.info()


df_train.info()


df = df_train.copy()


df.head()


unique_values_by_object_column = {}
for col_name in df.columns:
    if df[col_name].dtype == 'object':
        unique_values_by_object_column[col_name] = df[col_name].unique()

# Print the results
for col, unique_vals in unique_values_by_object_column.items():
    print(f"Unique values in object column '{col}': {unique_vals}")


#categorical_cols = ['gender', 'marital_status', 'education_level','employment_status', 'loan_purpose','grade_subgrade']

# # Initialize encoder
# encoder = OneHotEncoder(sparse=False)

# # Fit and transform
# encoded = encoder.fit_transform(df[categorical_cols])

# # Create DataFrame from encoded output
# encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))

# # Drop original categorical columns from main DataFrame
# df = df.drop(columns=categorical_cols)

# # Concatenate the encoded columns back
# df = pd.concat([df, encoded_df], axis=1)


df.columns


#corr = df.corr()


# mask = np.triu(np.ones_like(corr, dtype=bool))
# plt.figure(figsize=(24, 16))
# sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
# plt.title("Feature Correlation Heatmap (Lower Triangle)")
# plt.show()



df_loan_paid = df[(df['loan_paid_back']==0.4)]


df['loan_paid_back'].unique()


df_loan_paid


from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score


df.columns


X = df.drop(columns=['id','loan_paid_back'], errors='ignore')
y = df['loan_paid_back']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



cats = ['gender', 'marital_status', 'education_level',
        'employment_status', 'loan_purpose', 'grade_subgrade']

for c in cats:
    X_train[c] = X_train[c].astype('category')
    X_test[c]  = X_test[c].astype('category')
    
xgb_model = XGBRegressor(
    n_estimators=600,
    enable_categorical=True,
    learning_rate=0.02,   
    max_depth=3,  #6
    min_child_weight=3,
    subsample=0.8,        
    colsample_bytree=0.8,
    gamma=0.2,
    reg_alpha=0.3,
    reg_lambda=1.2,
    random_state=42    #42
)



xgb_model.fit(X_train, y_train,
              eval_set=[(X_train, y_train),(X_test, y_test)],
              eval_metric='rmse',
              verbose=False,
              early_stopping_rounds=200)


y_pred = xgb_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")


results = xgb_model.evals_result()

train_loss = results['validation_0']['rmse']
valid_loss = results['validation_1']['rmse']


import matplotlib.pyplot as plt

plt.plot(train_loss, label="Train RMSE")
plt.plot(valid_loss, label="Validation RMSE")
plt.legend()
plt.show()


cats = ['gender', 'marital_status', 'education_level',
        'employment_status', 'loan_purpose', 'grade_subgrade']

for c in cats:
    X_train[c] = X_train[c].astype('category')
    X_test[c]  = X_test[c].astype('category')
    
xgbClassifier = XGBClassifier(random_state=42,
                              objective='binary:logistic',
                              enable_categorical=True,
                             n_estimators=500,          # number of trees
                             learning_rate=0.06,        # shrinkage
                                max_depth=4,               # complexity of trees
                                subsample=0.8,             # row sampling
                            colsample_bytree=0.8,      # feature sampling
                                gamma=0,                   # regularization
                            reg_lambda=1,              # L2 regularization
                            reg_alpha=0,               # L1 regularization
                            min_child_weight=1,        # minimum leaf weight
                                eval_metric='auc' )  #logloss

xgbClassifier.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False,
              early_stopping_rounds=200)



y_pred = xgbClassifier.predict(X_test)
print("Accuracy of xbgClassifier:", accuracy_score(y_test, y_pred))



print(confusion_matrix(y_test, y_pred))



print(classification_report(y_test, y_pred))


from sklearn.metrics import roc_auc_score

y_prob = xgbClassifier.predict_proba(X_test)[:, 1]
print("AUC:", roc_auc_score(y_test, y_prob))


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from xgboost import XGBClassifier


df_test.info()


# Prediction on Test Dataset
test = df_test
test_ids = test['id'] 

# Drop ID before prediction
X_test_final = test.drop(columns=['id'])

cats = ['gender', 'marital_status', 'education_level',
        'employment_status', 'loan_purpose', 'grade_subgrade']

for c in cats:
    X_test_final[c]  = X_test_final[c].astype('category')


cats = ['gender', 'marital_status', 'education_level',
        'employment_status', 'loan_purpose', 'grade_subgrade']

# Convert types
for c in cats:
    X[c] = X[c].astype('category')

# Make a features-only copy of test (keep original df_test for ids)
df_test_features = df_test.drop(columns=["id"]).copy()
for c in cats:
    df_test_features[c] = df_test_features[c].astype('category')

# 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_predictions = np.zeros(len(X))
test_predictions = np.zeros(len(df_test))   # same number of rows as test
fold_scores = []

for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"Training fold {fold+1}...")
    
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    model = XGBClassifier(
        n_estimators=600,
        enable_categorical=True,
        learning_rate=0.02,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.2,
        reg_alpha=0.3,
        reg_lambda=1.2,
        random_state=42,
        # objective="reg:squarederror",
        eval_metric="logloss"
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
        early_stopping_rounds=100
    )
    
    # OOF preds
    oof_predictions[valid_idx] = model.predict_proba(X_valid)[:, 1]

    # average test preds over folds
    test_predictions += model.predict_proba(df_test_features)[:, 1] / kf.n_splits

    # Fold score
    fold_rmse = np.sqrt(np.mean((y_valid - oof_predictions[valid_idx])**2))
    fold_scores.append(fold_rmse)
    print(f"Fold {fold+1} RMSE: {fold_rmse}")

print("\nAverage RMSE:", np.mean(fold_scores))


# categorical_cols_test = ['gender', 'marital_status', 'education_level','employment_status', 'loan_purpose','grade_subgrade']

# # Initialize encoder
# encoder = OneHotEncoder(sparse=False)

# # Fit and transform
# encoded = encoder.fit_transform(df_test[categorical_cols_test])

# # Create DataFrame from encoded output
# encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols_test))

# # Drop original categorical columns from main DataFrame
# df_test = df_test.drop(columns=categorical_cols_test)

# # Concatenate the encoded columns back
# df_test = pd.concat([df_test, encoded_df], axis=1)


#X_test_final.info()





#submission of Regression model and Classification model
# y_pred_test = xgbClassifier.predict(X_test_final)
#y_pred_test1 = xgb_model.predict(X_test_final)



# xgbClassification model submission
# submission = pd.DataFrame({
#     'id': test_ids,
#     'loan_paid_back': y_pred_test
# })

# submission['loan_paid_back'] = submission['loan_paid_back'].clip(0, 1)
# submission.to_csv('submission.csv', index=False)


# xgbRegression model submission
# submission = pd.DataFrame({
#     'id': test_ids,
#     'loan_paid_back': y_pred_test1
# })

# submission['loan_paid_back'] = submission['loan_paid_back'].clip(0, 1)
# submission.to_csv('submission.csv', index=False)


# submission of K fold predictions
submission = pd.DataFrame()
submission["id"] = df_test["id"]
submission["loan_paid_back"] = test_predictions

submission.to_csv("submission.csv", index=False)
print("submission.csv created!")



submission.head()




