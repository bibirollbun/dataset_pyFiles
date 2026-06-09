# === Import lib ===
import numpy as np 
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns


# === Load Data ===
train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission_df = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


# === Explore Train Data ===
print("\n â˜‘ï¸� Data Info:")
train_df.info()
print("\n â˜‘ï¸� Numerical Features Summary:")
display(train_df.describe())
print("\n â˜‘ï¸� First 10 Rows of the Dataset:")
display(train_df.head(10))


# === Explore Test Data ===
print("\n â˜‘ï¸� Data Info:")
test_df.info()
print("\n â˜‘ï¸� Numerical Features Summary:")
display(test_df.describe())
print("\n â˜‘ï¸� First 10 Rows of the Dataset:")
display(test_df.head(10))


numerical_features = [
    "Age",
    "Height",
    "Weight",
    "Duration",
    "Heart_Rate",
    "Body_Temp",
    "Calories"
]
# Add a 'dataset' column to distinguish between train and test DataFrames
train_df['dataset'] = 'train'
test_df['dataset'] = 'test'

# Combine the DataFrames
combined_df = pd.concat([train_df, test_df], ignore_index=True)

for feature in numerical_features:
    plt.figure(figsize=(15, 6))

    # Histogram with KDE for training data
    plt.subplot(1, 2, 1)
    sns.histplot(combined_df[combined_df['dataset'] == 'train'][feature],
                 kde=True, bins=30, label='Train', color='skyblue')
    sns.histplot(combined_df[combined_df['dataset'] == 'test'][feature],
                 kde=True, bins=30, label='Test', color='salmon')
    plt.title(f"Distribution of {feature} (Train vs. Test)")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.legend()

    # Box plot comparing train and test data
    plt.subplot(1, 2, 2)
    sns.boxplot(x='dataset', y=feature, data=combined_df, palette={'train': 'skyblue', 'test': 'salmon'})
    plt.title(f"Box Plot of {feature} (Train vs. Test)")
    plt.xlabel("Dataset")
    plt.ylabel(feature)

    plt.tight_layout()
    plt.show()


colors = sns.color_palette('husl', 2)
rows = -(-len(numerical_features) // 4)
plt.figure(figsize=(20, 5 * rows))

for i, col in enumerate(numerical_features, 1):
    plt.subplot(rows, 4, i)
    sns.kdeplot(data=train_df[train_df['Sex'] == 'female'], x=col, fill=True, color=colors[0], label='Female')
    sns.kdeplot(data=train_df[train_df['Sex'] == 'male'], x=col, fill=True, color=colors[1], label='Male')
    plt.title(f'KDE Plot of {col} by Sex', fontsize=14)
    plt.xlabel(col)
    plt.ylabel('Density')
    plt.legend(title='Sex')

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

num_features_to_plot = len(numerical_features[:-1])
n_cols = 2
n_rows = (num_features_to_plot + n_cols - 1) // n_cols

fig_scatter, axes_scatter = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))
axes_scatter = axes_scatter.flatten()

colors = sns.color_palette('husl', 2)

for i, feature in enumerate(numerical_features[:-1]):
    sns.scatterplot(
        x=train_df[feature],
        y=train_df["Calories"],
        hue=train_df["Sex"],
        palette={'female': colors[0], 'male': colors[1]},
        alpha=0.5,
        ax=axes_scatter[i]
    )
    axes_scatter[i].set_title(f"{feature} vs. Calories by Sex")
    axes_scatter[i].set_xlabel(feature)
    axes_scatter[i].set_ylabel("Calories")
    axes_scatter[i].legend(title='Sex')

# Hide any unused subplots if the grid isn't perfectly filled
for j in range(i + 1, len(axes_scatter)):
    fig_scatter.delaxes(axes_scatter[j])

fig_scatter.tight_layout() 
plt.show()


correlation_matrix = train_df[numerical_features].corr()
plt.figure(figsize=(10, 8)) # Adjust the size here as needed
sns.heatmap(correlation_matrix, annot=True, cmap="viridis", fmt=".2f")
plt.title("Correlation Matrix of Numerical Features")
plt.show()


# === Encode SEX ===
# Encode 'Sex'
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_df['Sex'] = le.fit_transform(train_df['Sex'])
test_df['Sex'] = le.transform(test_df['Sex'])


# === Drop not needed columns ===
# Drop the 'dataset' column from train_df
train_df = train_df.drop(columns=['dataset'])
# Drop the 'dataset' column from test_df
test_df = test_df.drop(columns=['dataset'])


# Calculate BMI for the training data
train_df['BMI'] = train_df['Weight'] / (train_df['Height'] / 100)**2
# Calculate BMI for the testing data
test_df['BMI'] = test_df['Weight'] / (test_df['Height'] / 100)**2

baseline_temp = 37.0
# Calculate 'Temp_Change' for the training data
train_df['Temp_Change'] = train_df['Body_Temp'] - baseline_temp
# Calculate 'Temp_Change' for the testing data
test_df['Temp_Change'] = test_df['Body_Temp'] - baseline_temp

# Calculate 'Intensity' for the training data
train_df['Intensity'] = train_df['Heart_Rate'] / train_df['Duration']
# Calculate 'Intensity' for the testing data
test_df['Intensity'] = test_df['Heart_Rate'] / test_df['Duration']

# Calculate 'Heart_Rate_Ratio' for the training data
train_df['Heart_Rate_Ratio'] = train_df['Heart_Rate'] / train_df['Age']
# Calculate 'Heart_Rate_Ratio' for the testing data
test_df['Heart_Rate_Ratio'] = test_df['Heart_Rate'] / test_df['Age']

# Calculate 'Duration_x_HeartRate' for the training data
train_df['Duration_x_HeartRate'] = train_df['Duration'] * train_df['Heart_Rate']
# Calculate 'Duration_x_HeartRate' for the testing data
test_df['Duration_x_HeartRate'] = test_df['Duration'] * test_df['Heart_Rate']

# Calculate 'Weight_x_Duration' for the training data
train_df['Weight_x_Duration'] = train_df['Weight'] * train_df['Duration']
# Calculate 'Weight_x_Duration' for the testing data
test_df['Weight_x_Duration'] = test_df['Weight'] * test_df['Duration']

# Calculate 'Height_x_Duration' for the training data
train_df['Height_x_Duration'] = train_df['Height'] * train_df['Duration']
# Calculate 'Height_x_Duration' for the testing data
test_df['Height_x_Duration'] = test_df['Height'] * test_df['Duration']

# Calculate 'Weight_x_Height' for the training data
train_df['Weight_x_Height'] = train_df['Weight'] * train_df['Height']
# Calculate 'Weight_x_Height' for the testing data
test_df['Weight_x_Height'] = test_df['Weight'] * test_df['Height']

# Calculate 'Weight_x_Intensity' for the training data
train_df['Weight_x_Intensity'] = train_df['Weight'] * train_df['Intensity']
# Calculate 'Weight_x_Intensity' for the testing data
test_df['Weight_x_Intensity'] = test_df['Weight'] * test_df['Intensity']

# Calculate 'Height_x_Intensity' for the training data
train_df['Height_x_Intensity'] = train_df['Height'] * train_df['Intensity']
# Calculate 'Height_x_Intensity' for the testing data
test_df['Height_x_Intensity'] = test_df['Height'] * test_df['Intensity']

# === Adding new features ===

# Duration_squared
train_df['Duration_squared'] = train_df['Duration'] ** 2
test_df['Duration_squared'] = test_df['Duration'] ** 2

# Duration_x_Body_Temp
train_df['Duration_x_Body_Temp'] = train_df['Duration'] * train_df['Body_Temp']
test_df['Duration_x_Body_Temp'] = test_df['Duration'] * test_df['Body_Temp']

# Duration_log
train_df['Duration_log'] = np.log1p(train_df['Duration'])
test_df['Duration_log'] = np.log1p(test_df['Duration'])

# Heart_Rate_x_Body_Temp
train_df['Heart_Rate_x_Body_Temp'] = train_df['Heart_Rate'] * train_df['Body_Temp']
test_df['Heart_Rate_x_Body_Temp'] = test_df['Heart_Rate'] * test_df['Body_Temp']

# Duration_div_Body_Temp
train_df['Duration_div_Body_Temp'] = train_df['Duration'] / train_df['Body_Temp']
test_df['Duration_div_Body_Temp'] = test_df['Duration'] / test_df['Body_Temp']

# Heart_Rate_squared
train_df['Heart_Rate_squared'] = train_df['Heart_Rate'] ** 2
test_df['Heart_Rate_squared'] = test_df['Heart_Rate'] ** 2

# Age_x_Duration
train_df['Age_x_Duration'] = train_df['Age'] * train_df['Duration']
test_df['Age_x_Duration'] = test_df['Age'] * test_df['Duration']

# Age_x_Weight
train_df['Age_x_Weight'] = train_df['Age'] * train_df['Weight']
test_df['Age_x_Weight'] = test_df['Age'] * test_df['Weight']

# Age_x_Heart_Rate
train_df['Age_x_Heart_Rate'] = train_df['Age'] * train_df['Heart_Rate']
test_df['Age_x_Heart_Rate'] = test_df['Age'] * test_df['Heart_Rate']

# Heart_Rate_log
train_df['Heart_Rate_log'] = np.log1p(train_df['Heart_Rate'])
test_df['Heart_Rate_log'] = np.log1p(test_df['Heart_Rate'])

# Age_x_Height
train_df['Age_x_Height'] = train_df['Age'] * train_df['Height']
test_df['Age_x_Height'] = test_df['Age'] * test_df['Height']


# Display the first few rows of the DataFrames with the new Feature Engineering columns
print("\n â˜‘ï¸� Training data with Feature Engineering:")
display(train_df.info())
print("\n â˜‘ï¸� Testing data with Feature Engineering:")
display(test_df.info())


# Display the first few rows of the DataFrames with the new Feature Engineering columns
print("\n â˜‘ï¸� Training data with Feature Engineering:")
display(train_df.head(10))
print("\n â˜‘ï¸� Testing data with Feature Engineering:")
display(test_df.head(10))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_log_error
### -== Another meta-learner option ==- ###
from sklearn.linear_model import Ridge
#from sklearn.ensemble import RandomForestRegressor 


# Select all columns except 'id' and 'Calories' as features
features = [col for col in train_df.columns if col not in ['id', 'Calories']]
target = 'Calories'

X = train_df[features]
y = train_df[target]
X_test = test_df[features]

# --- Common Setup ---
# Function to calculate RMSLE
def rmsle(y_true, y_pred):
    return np.sqrt(np.mean(np.power(np.log1p(y_true) - np.log1p(y_pred), 2)))

# Initialize K-Fold for consistent splits
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("ğŸ”„ Starting Stacking Ensemble Preparation ğŸ”„")
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


# cat_params = {
#     'max_depth': 10,
#     'l2_leaf_reg': 2,
#     'learning_rate': 0.06,
#     'bagging_temperature': 0.08412755040615273,
#     'border_count': 222,
#     'loss_function': 'RMSE',
#     'random_state': 42,
#     'task_type': 'GPU',
#     'iterations': 3000
# }

# # XGBoost parameters
# xgb_params = {
#     'max_depth': 10,
#     'n_estimators': 3000,
#     'learning_rate': 0.07,
#     'gamma': 0.01,
#     'max_delta_step': 2,
#     'eval_metric': 'rmse',
#     'enable_categorical': False,
#     'random_state': 42,
#     'tree_method': 'gpu_hist',
#     'n_jobs': -1
# }

# # LGBM parameters
# lgbm_params = {
#     'objective': 'regression_l1',
#     'metric': 'rmse',
#     'n_estimators': 3000,
#     'learning_rate': 0.05,
#     'feature_fraction': 0.8,
#     'bagging_fraction': 0.8,
#     'bagging_freq': 1,
#     'lambda_l1': 0.1,
#     'lambda_l2': 0.1,
#     'num_leaves': 64,
#     'verbose': -1,
#     'n_jobs': -1,
#     'seed': 42,
#     'boosting_type': 'gbdt'
# }

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
print("ğŸ”„Generating Out-of-Fold (OOF) predictions and Test predictions for Base Models...")
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


print("\nâœ… Base Model OOF and Test Predictions Generated.")
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
submission_df = pd.DataFrame({'id': test_df['id'], 'Calories': final_predictions})
submission_df.to_csv('stacked_submission.csv', index=False)
print("\nâœ… Submission file 'stacked_submission.csv' created.")


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_log_error
import matplotlib.pyplot as plt
import seaborn as sns

# Select all columns except 'id' and 'Calories' as features
features = [col for col in train_df.columns if col not in ['id', 'Calories']]
target = 'Calories'

X = train_df[features]
y = train_df[target]
X_test = test_df[features]

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
submission_df = pd.DataFrame({'id': test_df['id'], 'Calories': final_averaged_predictions})
submission_df.to_csv('averaged_submission.csv', index=False)
print("\nâœ… Submission file 'averaged_submission.csv' created.")


# --- Manual Weighting for Final Averaged Predictions ---
print("\nğŸ”„ Applying manual weights for final predictions...")
print("--------------------------------------------------")

w_cat = 0.4 # === SET
w_xgb = 0.3  # === SET
w_lgbm = 0.3  # === SET

# Ensure weights sum to 1.0 (optional, but good practice for interpretability)
total_weight = w_cat + w_xgb + w_lgbm
if total_weight != 1.0:
    print(f"  âš ï¸� Warning: Weights do not sum to 1.0 (they sum to {total_weight:.2f}). Adjusting them proportionally.")
    w_cat /= total_weight
    w_xgb /= total_weight
    w_lgbm /= total_weight
    print(f"  Adjusted weights: CatBoost={w_cat:.2f}, XGBoost={w_xgb:.2f}, LGBM={w_lgbm:.2f}")

# Calculate the weighted average of the test predictions
final_weighted_predictions = (
    w_cat * test_preds_cat +
    w_xgb * test_preds_xgb +
    w_lgbm * test_preds_lgbm
)

# Ensure predictions are non-negative, as calorie expenditure cannot be negative
final_weighted_predictions[final_weighted_predictions < 0] = 0

print("âœ… Final Weighted Predictions Generated.")
print("--------------------------------------------------")
print("\nğŸš€ Weighted Averaging Ensemble Process Complete! ğŸš€")
print("ğŸ”� Example of final weighted predictions (first 5):")
print(final_weighted_predictions[:5])

# Save the submission DataFrame to a CSV file
submission_df = pd.DataFrame({'id': test_df['id'], 'Calories': final_weighted_predictions})
submission_df.to_csv('weighted_averaged_submission.csv', index=False)
print("\nâœ… Submission file 'weighted_averaged_submission.csv' created.")

