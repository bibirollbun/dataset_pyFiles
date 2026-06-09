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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from sklearn.model_selection import KFold
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import Ridge


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

sample_sub_df = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


print('â„¹ï¸� Sample Rows\n')
display(train.head(10))

print('\n\nâ„¹ï¸� Brief Information\n')
train.info()

print('\n\nâ„¹ï¸� Detailed Description on each Column\n')
display(train.describe(include = 'all').T)


print('â„¹ï¸� Sample Rows\n')
display(test.head(10))

print('\n\nâ„¹ï¸� Brief Information\n')
test.info()

print('\n\nâ„¹ï¸� Detailed Description on each Column\n')
display(test.describe(include = 'all').T)


#checking for duplicate records
train[train.duplicated()]


#check for null values
train.isnull().sum()


# check for data disctribution and outliers for numerical columns for train set

def num_distribution(df, numerical_features):
    for feature in numerical_features:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.histplot(df[feature], kde=True, bins=30)
        plt.title(f"Histogram of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")

        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[feature])
        plt.title(f"Box Plot of {feature}")

        plt.tight_layout()
        plt.show()

        print(f"\nStatistics for {feature}:")
        print(f"Skewness: {df[feature].skew():.2f}")
        print(f"Number of Missing Values: {df[feature].isnull().sum()}")


num_features = ["Age",
    "Height",
    "Weight",
    "Duration",
    "Heart_Rate",
    "Body_Temp",
    "Calories"]

num_distribution(train, num_features)




# check for data disctribution and outliers for numerical columns for test set

test_features = ["Age",
    "Height",
    "Weight",
    "Duration",
    "Heart_Rate",
    "Body_Temp"]


num_distribution(test, test_features)


def kde_plot(df, numerical_features):
    colors = sns.color_palette('husl', 2)
    rows = -(-len(numerical_features) // 4)
    plt.figure(figsize=(20, 5 * rows))

    for i, col in enumerate(numerical_features, 1):
        plt.subplot(rows, 4, i)
        sns.kdeplot(data=df[df['Sex'] == 'female'], x=col, fill=True, color=colors[0], label='Female')
        sns.kdeplot(data=df[df['Sex'] == 'male'], x=col, fill=True, color=colors[1], label='Male')
        plt.title(f'KDE Plot of {col} by Sex', fontsize=14)
        plt.xlabel(col)
        plt.ylabel('Density')
        plt.legend(title='Sex')

    plt.tight_layout()
    plt.show()

kde_plot(train, num_features)


correlation_matrix = train[num_features].corr()
plt.figure(figsize=(10, 8)) # Adjust the size here as needed
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Numerical Features")
plt.show()


# Encode 'Sex'
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])


# Calculate BMI for the training data
train['BMI'] = train['Weight'] / (train['Height'] / 100)**2
# Calculate BMI for the testing data
test['BMI'] = test['Weight'] / (test['Height'] / 100)**2

baseline_temp = 37.0
# Calculate 'Temp_Change' for the training data
train['Temp_Change'] = train['Body_Temp'] - baseline_temp
# Calculate 'Temp_Change' for the testing data
test['Temp_Change'] = test['Body_Temp'] - baseline_temp

# Calculate 'Intensity' for the training data
train['Intensity'] = train['Heart_Rate'] / train['Duration']
# Calculate 'Intensity' for the testing data
test['Intensity'] = test['Heart_Rate'] / test['Duration']

# Calculate 'Heart_Rate_Ratio' for the training data
train['Heart_Rate_Ratio'] = train['Heart_Rate'] / train['Age']
# Calculate 'Heart_Rate_Ratio' for the testing data
test['Heart_Rate_Ratio'] = test['Heart_Rate'] / test['Age']

# Calculate 'Duration_x_HeartRate' for the training data
train['Duration_x_HeartRate'] = train['Duration'] * train['Heart_Rate']
# Calculate 'Duration_x_HeartRate' for the testing data
test['Duration_x_HeartRate'] = test['Duration'] * test['Heart_Rate']

# Calculate 'Weight_x_Duration' for the training data
train['Weight_x_Duration'] = train['Weight'] * train['Duration']
# Calculate 'Weight_x_Duration' for the testing data
test['Weight_x_Duration'] = test['Weight'] * test['Duration']

# Calculate 'Height_x_Duration' for the training data
train['Height_x_Duration'] = train['Height'] * train['Duration']
# Calculate 'Height_x_Duration' for the testing data
test['Height_x_Duration'] = test['Height'] * test['Duration']

# Calculate 'Weight_x_Height' for the training data
train['Weight_x_Height'] = train['Weight'] * train['Height']
# Calculate 'Weight_x_Height' for the testing data
test['Weight_x_Height'] = test['Weight'] * test['Height']

# Calculate 'Weight_x_Intensity' for the training data
train['Weight_x_Intensity'] = train['Weight'] * train['Intensity']
# Calculate 'Weight_x_Intensity' for the testing data
test['Weight_x_Intensity'] = test['Weight'] * test['Intensity']

# Calculate 'Height_x_Intensity' for the training data
train['Height_x_Intensity'] = train['Height'] * train['Intensity']
# Calculate 'Height_x_Intensity' for the testing data
test['Height_x_Intensity'] = test['Height'] * test['Intensity']


# Duration_squared
train['Duration_squared'] = train['Duration'] ** 2
test['Duration_squared'] = test['Duration'] ** 2

# Duration_x_Body_Temp
train['Duration_x_Body_Temp'] = train['Duration'] * train['Body_Temp']
test['Duration_x_Body_Temp'] = test['Duration'] * test['Body_Temp']

# Duration_log
train['Duration_log'] = np.log1p(train['Duration'])
test['Duration_log'] = np.log1p(test['Duration'])

# Heart_Rate_x_Body_Temp
train['Heart_Rate_x_Body_Temp'] = train['Heart_Rate'] * train['Body_Temp']
test['Heart_Rate_x_Body_Temp'] = test['Heart_Rate'] * test['Body_Temp']

# Duration_div_Body_Temp
train['Duration_div_Body_Temp'] = train['Duration'] / train['Body_Temp']
test['Duration_div_Body_Temp'] = test['Duration'] / test['Body_Temp']

# Heart_Rate_squared
train['Heart_Rate_squared'] = train['Heart_Rate'] ** 2
test['Heart_Rate_squared'] = test['Heart_Rate'] ** 2

# Age_x_Duration
train['Age_x_Duration'] = train['Age'] * train['Duration']
test['Age_x_Duration'] = test['Age'] * test['Duration']

# Age_x_Weight
train['Age_x_Weight'] = train['Age'] * train['Weight']
test['Age_x_Weight'] = test['Age'] * test['Weight']

# Age_x_Heart_Rate
train['Age_x_Heart_Rate'] = train['Age'] * train['Heart_Rate']
test['Age_x_Heart_Rate'] = test['Age'] * test['Heart_Rate']

# Heart_Rate_log
train['Heart_Rate_log'] = np.log1p(train['Heart_Rate'])
test['Heart_Rate_log'] = np.log1p(test['Heart_Rate'])

# Age_x_Height
train['Age_x_Height'] = train['Age'] * train['Height']
test['Age_x_Height'] = test['Age'] * test['Height']


# Removing columns 'id' and 'Calories' 
features = [col for col in train.columns if col not in ['id', 'Calories']]
target = 'Calories'

X = train[features]
y = train[target]
X_test = test[features]

# Function to calculate RMSLE
def rmsle(y_true, y_pred):
    return np.sqrt(np.mean(np.power(np.log1p(y_true) - np.log1p(y_pred), 2)))

# Initialize K-Fold for consistent splits
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("Starting Ensemble Preparation ")
print("---------------------------------------------")

# --- Model Parameters ---
# CatBoost parameters 
cat_params = {
    'max_depth': 10,
    'l2_leaf_reg': 2,
    'learning_rate': 0.06,
    'bagging_temperature': 0.08412755040615273,
    'border_count': 222,
    'loss_function': 'RMSE',
    'random_state': 42,
    'task_type': 'GPU',
    'iterations': 3000
}

# XGBoost parameters
xgb_params = {
    'n_estimators': 2642,
    'learning_rate': 0.013787764619353767,
    'max_depth': 5,
    'min_child_weight': 4,
    'gamma': 4.452048365748842e-05,
    'subsample': 0.9140703845572055,
    'colsample_bytree': 0.6798695128633439,
    'lambda': 0.00042472707398058225,
    'alpha': 0.0021465011216654484,
    'max_delta_step': 0,
    #'eval_metric': 'rmse',
    'enable_categorical': False,
    'random_state': 42,
    'tree_method': 'gpu_hist',
    'n_jobs': -1 
}

# LGBM parameters (kept as in original Code 2)
lgbm_params = {
    'n_estimators': 2978,
    'learning_rate': 0.010485387725194618,
    'num_leaves': 250,
    'max_depth': 11,
    'min_child_samples': 37,
    'subsample': 0.6727299868828402,
    'colsample_bytree': 0.6733618039413735,
    'lambda_l1': 5.472429642032198e-06,
    'lambda_l2': 0.00052821153945323,
    #'objective': 'regression_l1',
    #'metric': 'rmse', 
    'verbose': -1, 
    'n_jobs': -1,
    'seed': 42,
    'boosting_type': 'gbdt'
}


# --- Prediction Storage ---
# Arrays to store out-of-fold (OOF)
oof_preds_cat = np.zeros(len(X))
oof_preds_xgb = np.zeros(len(X))
oof_preds_lgbm = np.zeros(len(X))

# Arrays to store test predictions (accumulated across folds for averaging)
test_preds_cat = np.zeros(len(X_test))
test_preds_xgb = np.zeros(len(X_test))
test_preds_lgbm = np.zeros(len(X_test))

# --- Cross-Validation Loop for Base Models ---
print("Generating Out-of-Fold (OOF) predictions and Test predictions for Base Models...")
print("-----------------------------------------------------------------------")

for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
    print(f"\n ğŸš€ --- Fold {fold+1}/{kf.n_splits} ---")

    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Apply log1p transformation to target for training
    y_train_log1p = np.log1p(y_train)
    y_val_log1p = np.log1p(y_val)

    # --- CatBoost Training and Prediction ---
    print("  â�¡ Training CatBoost...")
    cat_model = CatBoostRegressor(**cat_params)
    cat_model.fit(X_train, y_train_log1p,
                  eval_set=[(X_val, y_val_log1p)],
                  early_stopping_rounds=200,
                  verbose=0 # Set to 100 if you want to see progress
                 )
    cat_val_pred_log1p = cat_model.predict(X_val)
    cat_test_pred_log1p = cat_model.predict(X_test)

    # --- XGBoost Training and Prediction ---
    print("  â�¡ Training XGBoost...")
    xgb_model = XGBRegressor(**xgb_params) 
    xgb_model.fit(X_train, y_train_log1p,
                  eval_set=[(X_val, y_val_log1p)],
                  early_stopping_rounds=200,
                  verbose=False # Set to 100 if you want to see progress
                 )
    xgb_val_pred_log1p = xgb_model.predict(X_val)
    xgb_test_pred_log1p = xgb_model.predict(X_test)

    # --- LGBM Training and Prediction ---
    print("  â�¡ Training LGBM...")
    lgbm_model = lgb.LGBMRegressor(**lgbm_params) # Re-initialize for each fold
    lgbm_model.fit(X_train, y_train_log1p,
                   eval_set=[(X_val, y_val_log1p)],
                   eval_metric='rmse',
                   callbacks=[lgb.log_evaluation(period=0), lgb.early_stopping(stopping_rounds=200, verbose=False)]
                  )
    lgbm_val_pred_log1p = lgbm_model.predict(X_val)
    lgbm_test_pred_log1p = lgbm_model.predict(X_test)

    # --- Store OOF and Test Predictions (transformed back to original scale) ---
    oof_preds_cat[val_index] = np.expm1(cat_val_pred_log1p)
    oof_preds_xgb[val_index] = np.expm1(xgb_val_pred_log1p)
    oof_preds_lgbm[val_index] = np.expm1(lgbm_val_pred_log1p)

    test_preds_cat += np.expm1(cat_test_pred_log1p) / kf.n_splits
    test_preds_xgb += np.expm1(xgb_test_pred_log1p) / kf.n_splits
    test_preds_lgbm += np.expm1(lgbm_test_pred_log1p) / kf.n_splits

    # Ensure all predictions are non-negative
    oof_preds_cat[oof_preds_cat < 0] = 0
    oof_preds_xgb[oof_preds_xgb < 0] = 0
    oof_preds_lgbm[oof_preds_lgbm < 0] = 0
    # Note: test predictions will also be non-negative after final prediction step

    # Calculate and print RMSLE for individual models on this fold
    print(f"  âœ”ï¸�CatBoost RMSLE (Fold {fold+1}): {rmsle(y_val, oof_preds_cat[val_index]):.4f}")
    print(f"  âœ”ï¸�XGBoost RMSLE (Fold {fold+1}): {rmsle(y_val, oof_preds_xgb[val_index]):.4f}")
    print(f"  âœ”ï¸�LGBM RMSLE (Fold {fold+1}): {rmsle(y_val, oof_preds_lgbm[val_index]):.4f}")


print("\n Base Model OOF and Test Predictions Generated.")
print("--------------------------------------------------")

# --- 2. Prepare Data for Meta-Learner ---
print("ğŸ”„ Preparing data for the Meta-Learner...")

# Create the training dataset for the meta-learner using OOF predictions
X_meta_train = pd.DataFrame({
    'cat_pred': oof_preds_cat,
    'xgb_pred': oof_preds_xgb,
    'lgbm_pred': oof_preds_lgbm
})

# Create the test dataset for the meta-learner using averaged test predictions
X_meta_test = pd.DataFrame({
    'cat_pred': test_preds_cat,
    'xgb_pred': test_preds_xgb,
    'lgbm_pred': test_preds_lgbm
})

y_meta_train = y # The original target for the meta-learner

print(f"â�¡ Meta-training data shape: {X_meta_train.shape}")
print(f"â�¡ Meta-test data shape: {X_meta_test.shape}")
print("--------------------------------------------------")

# --- 3. Train the Meta-Learner ---
print("ğŸ”„ Training the Meta-Learner...")

# USE ONE of effective meta-learners
meta_model = Ridge(random_state=42)
#meta_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)

meta_model.fit(X_meta_train, y_meta_train)

print("âœ… Meta-Learner Trained.")
print("--------------------------------------------------")

# --- 4. Make Final Predictions ---
print("ğŸ”„ Making final predictions using the Meta-Learner...")

final_predictions = meta_model.predict(X_meta_test)

# Ensure predictions are non-negative, as calorie expenditure cannot be negative
final_predictions[final_predictions < 0] = 0

print("âœ… Final Predictions Generated.")
print("--------------------------------------------------")

print("\nğŸš€ Stacking Ensemble Process Complete! ğŸš€")
print("ğŸ”� Example of final predictions (first 5):")
print(final_predictions[:5])

# Evaluate the performance of the meta-learner on its training data

meta_train_preds = meta_model.predict(X_meta_train)
stacked_rmsle = rmsle(y_meta_train, meta_train_preds)
print(f"\nâœ… Stacked Model (Meta-Learner) RMSLE on OOF data: {stacked_rmsle:.4f}")

# Save the submission DataFrame to a CSV file
submission = pd.DataFrame({'id': test['id'], 'Calories': final_predictions})
submission.to_csv('stacked_submission.csv', index=False)
print("\nâœ… Submission file 'stacked_submission.csv' created.")





# Average model
# Select all columns except 'id' and 'Calories' as features
features = [col for col in train.columns if col not in ['id', 'Calories']]
target = 'Calories'

X = train[features]
y = train[target]
X_test = test[features]

# --- Common Setup ---
# Function to calculate RMSLE
def rmsle(y_true, y_pred):
    return np.sqrt(np.mean(np.power(np.log1p(y_true) - np.log1p(y_pred), 2)))

# Initialize K-Fold for consistent splits
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("ğŸ”„ Starting Individual Model Training and Averaging Preparation ğŸ”„")
print("-----------------------------------------------------------------")

# --- Model Parameters ---
# CatBoost parameters (kept as in original Code 2)
cat_params = {
    'max_depth': 10,
    'l2_leaf_reg': 2,
    'learning_rate': 0.06,
    'bagging_temperature': 0.08412755040615273,
    'border_count': 222,
    'loss_function': 'RMSE',
    'random_state': 42,
    'task_type': 'GPU',
    'iterations': 3000
}

# XGBoost parameters (kept as in original Code 2)
xgb_params = {
    'n_estimators': 2642,
    'learning_rate': 0.013787764619353767,
    'max_depth': 5,
    'min_child_weight': 4,
    'gamma': 4.452048365748842e-05,
    'subsample': 0.9140703845572055,
    'colsample_bytree': 0.6798695128633439,
    'lambda': 0.00042472707398058225,
    'alpha': 0.0021465011216654484,
    'max_delta_step': 0,
    'eval_metric': 'rmse', ###
    'enable_categorical': False,
    'random_state': 42,
    'tree_method': 'gpu_hist',
    'n_jobs': -1 
}

# LGBM parameters (kept as in original Code 2)
lgbm_params = {
    'n_estimators': 2978,
    'learning_rate': 0.010485387725194618,
    'num_leaves': 250,
    'max_depth': 11,
    'min_child_samples': 37,
    'subsample': 0.6727299868828402,
    'colsample_bytree': 0.6733618039413735,
    'lambda_l1': 5.472429642032198e-06,
    'lambda_l2': 0.00052821153945323,
    'objective': 'regression_l1', ###
    'metric': 'rmse', ###
    'verbose': -1, 
    'n_jobs': -1,
    'seed': 42,
    'boosting_type': 'gbdt'
}

# --- Prediction Storage ---
oof_cat = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_lgbm = np.zeros(len(X))

test_preds_cat = np.zeros(len(X_test))
test_preds_xgb = np.zeros(len(X_test))
test_preds_lgbm = np.zeros(len(X_test))

# --- Cross-Validation Loop ---
print("ğŸ”„ Generating Out-of-Fold (OOF) predictions and Test predictions for Base Models...")
print("-----------------------------------------------------------------------")

for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
    print(f"\n ğŸš€ --- Fold {fold+1}/{kf.n_splits} ---")

    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Apply log1p transformation to target for training
    y_train_log1p = np.log1p(y_train)
    y_val_log1p = np.log1p(y_val)

    # --- CatBoost Training ---
    print("  â�¡ Training CatBoost...")
    cat_model = CatBoostRegressor(**cat_params)
    cat_model.fit(X_train, y_train_log1p,
                  eval_set=[(X_val, y_val_log1p)],
                  early_stopping_rounds=200,
                  verbose=0 # Matches Code 1 verbose
                 )
    cat_val_pred_log1p = cat_model.predict(X_val)
    cat_test_pred_log1p = cat_model.predict(X_test)

    # --- XGBoost Training ---
    print("  â�¡ Training XGBoost...")
    xgb_model = XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train_log1p,
                  eval_set=[(X_val, y_val_log1p)],
                  early_stopping_rounds=200,
                  verbose=False # Matches Code 1 verbose
                 )
    xgb_val_pred_log1p = xgb_model.predict(X_val)
    xgb_test_pred_log1p = xgb_model.predict(X_test)

     # --- LGBM Training ---
    print("  â�¡ Training LGBM...")
    lgbm_model = lgb.LGBMRegressor(**lgbm_params)
    lgbm_model.fit(X_train, y_train_log1p,
                   eval_set=[(X_val, y_val_log1p)],
                   eval_metric='rmse',
                   callbacks=[lgb.log_evaluation(period=0), lgb.early_stopping(stopping_rounds=200, verbose=False)] # Matches Code 1 verbose
                  )
    lgbm_val_pred_log1p = lgbm_model.predict(X_val)
    lgbm_test_pred_log1p = lgbm_model.predict(X_test)

    # --- Store OOF and Test Predictions ---
    oof_cat[val_index] = np.expm1(cat_val_pred_log1p)
    oof_xgb[val_index] = np.expm1(xgb_val_pred_log1p)
    oof_lgbm[val_index] = np.expm1(lgbm_val_pred_log1p)

    test_preds_cat += np.expm1(cat_test_pred_log1p) / kf.n_splits
    test_preds_xgb += np.expm1(xgb_test_pred_log1p) / kf.n_splits
    test_preds_lgbm += np.expm1(lgbm_test_pred_log1p) / kf.n_splits

    # Ensure all predictions are non-negative
    oof_cat[oof_cat < 0] = 0
    oof_xgb[oof_xgb < 0] = 0
    oof_lgbm[oof_lgbm < 0] = 0

    # Calculate and print RMSLE for individual models on this fold
    print(f"  âœ”ï¸�CatBoost RMSLE (Fold {fold+1}): {rmsle(y_val, oof_cat[val_index]):.4f}")
    print(f"  âœ”ï¸�XGBoost RMSLE (Fold {fold+1}): {rmsle(y_val, oof_xgb[val_index]):.4f}")
    print(f"  âœ”ï¸�LGBM RMSLE (Fold {fold+1}): {rmsle(y_val, oof_lgbm[val_index]):.4f}")

print("\nâœ… Base Model OOF and Test Predictions Generated.")
print("--------------------------------------------------")

# --- Overall RMSLE for Individual Models (using OOF predictions) ---
print("\n--- Overall Individual Model RMSLE (OOF Predictions) ---")
overall_rmsle_cat = rmsle(y, oof_cat)
overall_rmsle_xgb = rmsle(y, oof_xgb)
overall_rmsle_lgbm = rmsle(y, oof_lgbm)
print(f"Overall CatBoost OOF RMSLE: {overall_rmsle_cat:.4f}")
print(f"Overall XGBoost OOF RMSLE: {overall_rmsle_xgb:.4f}")
print(f"Overall LGBM OOF RMSLE: {overall_rmsle_lgbm:.4f}")

# --- Overall RMSLE for the Averaged Ensemble (using OOF predictions) ---
print("\nâœ… Averaged Ensemble RMSLE on OOF data:")
averaged_oof_preds = (oof_cat + oof_xgb + oof_lgbm) / 3

# Ensure OOF predictions are non-negative before calculating RMSLE
averaged_oof_preds[averaged_oof_preds < 0] = 0
stacked_rmsle_oof = rmsle(y, averaged_oof_preds)
print(f"Overall Averaged Ensemble OOF RMSLE: {stacked_rmsle_oof:.4f}")
print("--------------------------------------------------")

# --- Final Averaged Predictions ---
print("\nğŸ”„ Generating final averaged predictions...")
final_averaged_predictions = (test_preds_cat + test_preds_xgb + test_preds_lgbm) / 3

# Ensure predictions are non-negative
final_averaged_predictions[final_averaged_predictions < 0] = 0

print("âœ… Final Averaged Predictions Generated.")
print("--------------------------------------------------")
print("\nğŸš€ Averaging Ensemble Process Complete! ğŸš€")
print("ğŸ”� Example of final predictions (first 5):")
print(final_averaged_predictions[:5])

# Save the submission DataFrame to a CSV file
submission_df = pd.DataFrame({'id': test['id'], 'Calories': final_averaged_predictions})
submission_df.to_csv('averaged_submission.csv', index=False)
print("\nâœ… Submission file 'averaged_submission.csv' created.")




