import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.simplefilter('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
print("Train Shape:", train.shape)
print("Test Shape :", test.shape)
train.head(3)


print(f"Train has {train.isnull().sum().sum()} null values")
print(f"Test has {test.isnull().sum().sum()} null values")


print(f"Train has {train.isnull().sum().sum()} null values")
print(f"Test has {test.isnull().sum().sum()} null values")

print("Null values in each column for train:")
print(train.isnull().sum())

print("\nNull values in each column for test:")
print(test.isnull().sum())


num_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 
            'Guest_Popularity_percentage', 'Number_of_Ads', 'Listening_Time_minutes']

cat_cols = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']

# Visualize continuous variables with histograms
for col in num_cols:
    plt.figure(figsize=(8, 4))
    # Drop missing values for the histogram
    plt.hist(train[col].dropna(), bins=30, alpha=0.7)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# Visualize categorical variables with bar charts
for col in cat_cols:
    plt.figure(figsize=(8, 4))
    value_counts = train[col].value_counts()
    value_counts.plot(kind='bar')
    plt.title(f'Value Counts for {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.show()


import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge


target = 'Listening_Time_minutes'
drop_cols = ['id', target] 
X = train.drop(columns=drop_cols)
y = train[target]
test = test.drop(columns='id')


%%time
# Define categorical columns
cat_features = ['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']

# Create 10-fold cross validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Arrays to hold out-of-fold predictions for stacking (for training data)
oof_xgb = np.zeros(len(X))
oof_lgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))

# Lists to store RMSE scores for each fold
scores_xgb = []
scores_lgb = []
scores_cat = []
scores_avg = []

# Lists to store test predictions for each fold for each model
test_preds_xgb = []
test_preds_lgb = []
test_preds_cat = []

for train_index, valid_index in kf.split(X):
    # Split data into training and validation
    X_train = X.iloc[train_index].copy()
    X_valid = X.iloc[valid_index].copy()
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    
    # ---------------------
    # Prepare data for XGBoost: convert categorical columns to 'category'
    X_train_xgb = X_train.copy()
    X_valid_xgb = X_valid.copy()
    for col in cat_features:
        if col in X_train_xgb.columns:
            X_train_xgb[col] = X_train_xgb[col].astype('category')
        if col in X_valid_xgb.columns:
            X_valid_xgb[col] = X_valid_xgb[col].astype('category')
    
    # ---------------------
    # Prepare data for LightGBM: convert categorical columns to 'category'
    X_train_lgb = X_train.copy()
    X_valid_lgb = X_valid.copy()
    for col in cat_features:
        if col in X_train_lgb.columns:
            X_train_lgb[col] = X_train_lgb[col].astype('category')
        if col in X_valid_lgb.columns:
            X_valid_lgb[col] = X_valid_lgb[col].astype('category')
            
    # ---------------------
    # Prepare data for CatBoost: convert categorical columns to 'str'
    X_train_cat = X_train.copy()
    X_valid_cat = X_valid.copy()
    for col in cat_features:
        if col in X_train_cat.columns:
            X_train_cat[col] = X_train_cat[col].astype('str')
        if col in X_valid_cat.columns:
            X_valid_cat[col] = X_valid_cat[col].astype('str')
    
    # ---------------------
    # Prepare test data for each model
    # For XGBoost and LightGBM: convert categorical columns to 'category'
    X_test_xgb = test.copy()
    X_test_lgb = test.copy()
    for col in cat_features:
        if col in X_test_xgb.columns:
            X_test_xgb[col] = X_test_xgb[col].astype('category')
        if col in X_test_lgb.columns:
            X_test_lgb[col] = X_test_lgb[col].astype('category')
    
    # For CatBoost: convert categorical columns to 'str'
    X_test_cat = test.copy()
    for col in cat_features:
        if col in X_test_cat.columns:
            X_test_cat[col] = X_test_cat[col].astype('str')
    
    # ---------------------
    # XGBoost 
    model_xgb = xgb.XGBRegressor(
        random_state=42,
        n_estimators=10000,
        learning_rate=0.05,
        max_depth=6,
        colsample_bytree=0.8,
        subsample=0.8,
        enable_categorical=True,
        device='cuda',
    )
    model_xgb.fit(
        X_train_xgb, y_train, 
        eval_set=[(X_valid_xgb, y_valid)],
        early_stopping_rounds=100,
        verbose=False
    )
    pred_xgb = model_xgb.predict(X_valid_xgb)
    oof_xgb[valid_index] = pred_xgb
    rmse_xgb = np.sqrt(mean_squared_error(y_valid, pred_xgb))
    scores_xgb.append(rmse_xgb)
    
    # Predict on test data with XGBoost
    test_pred_xgb = model_xgb.predict(X_test_xgb)
    test_preds_xgb.append(test_pred_xgb)
    
    # ---------------------
    # LightGBM
    model_lgb = lgb.LGBMRegressor(
        random_state=42,
        n_estimators=10000,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        learning_rate=0.05,
        device='gpu',
    )
    model_lgb.fit(
        X_train_lgb, y_train, 
        eval_set=[(X_valid_lgb, y_valid)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(5000)],
    )
    pred_lgb = model_lgb.predict(X_valid_lgb)
    oof_lgb[valid_index] = pred_lgb
    rmse_lgb = np.sqrt(mean_squared_error(y_valid, pred_lgb))
    scores_lgb.append(rmse_lgb)
    
    # Predict on test data with LightGBM
    test_pred_lgb = model_lgb.predict(X_test_lgb)
    test_preds_lgb.append(test_pred_lgb)
    
    # ---------------------
    # CatBoost Model
    model_cat = CatBoostRegressor(
        random_state=42,
        iterations=10000,
        learning_rate=0.05,
        od_type='Iter',
        od_wait=100,
        verbose=False,
        task_type='GPU',
    )
    model_cat.fit(
        X_train_cat, y_train,
        eval_set=(X_valid_cat, y_valid),
        cat_features=cat_features
    )
    pred_cat = model_cat.predict(X_valid_cat)
    oof_cat[valid_index] = pred_cat
    rmse_cat = np.sqrt(mean_squared_error(y_valid, pred_cat))
    scores_cat.append(rmse_cat)
    
    # Predict on test data with CatBoost
    test_pred_cat = model_cat.predict(X_test_cat)
    test_preds_cat.append(test_pred_cat)
    
    # ---------------------
    # Simple Average Ensemble for the current fold
    pred_avg = (pred_xgb + pred_lgb + pred_cat) / 3
    rmse_avg = np.sqrt(mean_squared_error(y_valid, pred_avg))
    scores_avg.append(rmse_avg)

# Print CV RMSE scores for each model and the simple average ensemble
print("XGBoost CV RMSE per fold:", scores_xgb)
print("LightGBM CV RMSE per fold:", scores_lgb)
print("CatBoost CV RMSE per fold:", scores_cat)
print("Simple Average Ensemble CV RMSE per fold:", scores_avg)

# ---------------------
# Average test predictions across folds for each model
test_preds_xgb_final = np.mean(np.vstack(test_preds_xgb), axis=0)
test_preds_lgb_final = np.mean(np.vstack(test_preds_lgb), axis=0)
test_preds_cat_final = np.mean(np.vstack(test_preds_cat), axis=0)

# Simple Average Ensemble on test predictions
test_preds_avg = (test_preds_xgb_final + test_preds_lgb_final + test_preds_cat_final) / 3

# ---------------------
# Ridge Stacking Ensemble for test predictions
# Create meta features for training from out-of-fold predictions
X_meta = np.column_stack([oof_xgb, oof_lgb, oof_cat])
# Train meta model on the full training meta-features
meta_model = Ridge(random_state=42)
meta_model.fit(X_meta, y)
# Create meta features for test data from the averaged predictions
X_meta_test = np.column_stack([test_preds_xgb_final, test_preds_lgb_final, test_preds_cat_final])
test_preds_stack = meta_model.predict(X_meta_test)


# Load the sample submission file
sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')

# XGBoost submission
sub[target] = test_preds_xgb_final
print("XGBoost Submission Preview:")
print(sub.head(3))
sub.to_csv("submission_xgboost.csv", index=False)

# LightGBM submission
sub[target] = test_preds_lgb_final
print("LightGBM Submission Preview:")
print(sub.head(3))
sub.to_csv("submission_lightgbm.csv", index=False)

# CatBoost submission
sub[target] = test_preds_cat_final
print("CatBoost Submission Preview:")
print(sub.head(3))
sub.to_csv("submission_catboost.csv", index=False)

# Simple Average Ensemble submission
sub[target] = test_preds_avg
print("Simple Average Ensemble Submission Preview:")
print(sub.head(3))
sub.to_csv("submission_simple_average.csv", index=False)

# Ridge Stacking Ensemble submission
sub[target] = test_preds_stack
print("Ridge Stacking Ensemble Submission Preview:")
print(sub.head(3))
sub.to_csv("submission_ridge_stacking.csv", index=False)

