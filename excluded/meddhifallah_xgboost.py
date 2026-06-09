import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score, f1_score
from sklearn.model_selection import GridSearchCV

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

train_file_path = r"/kaggle/input/become-a-kaggle-master-hw-1-2025/train.csv"

df = pd.read_csv(train_file_path)

print("First five rows:")
print(df.head())

print("\nDataset info:")
print(df.info())

print("\nDescriptive statistics:")
print(df.describe())


missing_values = df.isnull().sum()
print("Missing values per column:")
print(missing_values[missing_values > 0])

df['var5'] = df['var5'].fillna(df['var5'].median())
df['var10'] = df['var10'].fillna(df['var10'].median())

print("\nMissing values after imputation:")
print(df[['var5', 'var10']].isnull().sum())


X = df.drop(columns=['ID', 'TARGET'])
y = df['TARGET']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)


xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.03, 0.1],
    'subsample': [0.7, 0.8, 1],
    'colsample_bytree': [0.7, 0.8, 1],
    'reg_alpha': [0, 1],
    'reg_lambda': [0, 1],
    'random_state': [42]
}

xgb_model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',  
    use_label_encoder=False
)

xgb_grid = GridSearchCV(
    estimator=xgb_model,
    param_grid=xgb_param_grid,
    scoring='neg_mean_squared_error',
    cv=3,
    n_jobs=-1,
    verbose=1
)

xgb_grid.fit(X_train, y_train)

print("Best XGBoost params:", xgb_grid.best_params_)
print("Best XGBoost CV score (MSE):", -xgb_grid.best_score_)

best_xgb_model = xgb_grid.best_estimator_
best_xgb_model.fit(X_train, y_train)

y_val_pred_xgb = best_xgb_model.predict_proba(X_val)[:, 1]

mse_xgb = mean_squared_error(y_val, y_val_pred_xgb)
auc_xgb = roc_auc_score(y_val, y_val_pred_xgb)
f1_xgb = f1_score(y_val, (y_val_pred_xgb >= 0.5).astype(int))

print("\nValidation Performance of Best XGBoost Model:")
print("MSE:", mse_xgb)
print("AUC:", auc_xgb)
print("F1 Score:", f1_xgb)



X_full = df.drop(columns=['ID', 'TARGET'])
y_full = df['TARGET']

best_xgb_model.fit(X_full, y_full)

test_file_path = r"/kaggle/input/become-a-kaggle-master-hw-1-2025/test.csv"
test_df = pd.read_csv(test_file_path)

test_df['var5'] = test_df['var5'].fillna(test_df['var5'].median())
test_df['var10'] = test_df['var10'].fillna(test_df['var10'].median())

X_test = test_df.drop(columns=['ID'])

y_test_pred = best_xgb_model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    'ID': test_df['ID'],
    'TARGET': y_test_pred
})

submission.to_csv('submission_xgb_best.csv', index=False)
print("Submission file 'submission_xgb_best.csv' created.")


