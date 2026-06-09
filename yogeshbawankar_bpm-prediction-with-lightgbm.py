import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
import gc

# Configure settings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# Load data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')

# Visualize train df and test df
print("========= Train DF ==========")
train_df.head()



print("========= Test DF ==========")
test_df.head()


# Save the ID from test set for the final submission file 
test_ids = test_df['id']

# Save the target feature from the training set 
target = train_df['BeatsPerMinute']

# Store the number of training rows to split on later
n_train = len(train_df)

# Drop the columns we don't need for training 
train_df = train_df.drop(columns=['id','BeatsPerMinute'])
test_df = test_df.drop(columns=['id'])

# Concatenate them into a single dataframe 
combined_df = pd.concat([train_df,test_df],ignore_index=True)

print(f"Shape of original training data: {train_df.shape}")
print(f"Shape of original test data: {test_df.shape}")
print(f"Shape of combined data: {combined_df.shape}")


# copy of the main dataset for EDA
train_eda_df = combined_df.iloc[:n_train].copy()
train_eda_df['BeatsPerMinute'] = target

print("======== Statistical Summary ========")
styled_summary = train_eda_df.describe().T.style.background_gradient(cmap='viridis', low=0.2, high=0.8)
display(styled_summary)


# Univariate Analysis

# Plotting the distribution of all features in the combined dataframe
print("Feature distributions...")
features = combined_df.columns
num_features = len(features)
num_cols = 3
num_rows = (num_features + num_cols - 1) // num_cols

fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 12))
axes = axes.flatten()

for i, feature in enumerate(features):
    sns.histplot(combined_df[feature], ax=axes[i], kde=True, color=sns.color_palette("viridis", num_features)[i])
    axes[i].set_title(f'Distribution of {feature}', fontsize=12)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle('Feature Distributions in Combined Data', fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# Bivariate Analysis
plt.figure(figsize=(14, 10))
correlation_matrix = train_eda_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation Matrix Heatmap', fontsize=18)
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Energy vs BeatsPerMinute
sns.regplot(data=train_eda_df.sample(2000), x='Energy', y='BeatsPerMinute', 
            ax=axes[0], scatter_kws={'alpha':0.4, 'color':'#FF5733'}, line_kws={'color':'black'})
axes[0].set_title('Energy vs. BeatsPerMinute', fontsize=15)

# AudioLoudness vs BeatsPerMinute
sns.regplot(data=train_eda_df.sample(2000), x='AudioLoudness', y='BeatsPerMinute', 
            ax=axes[1], scatter_kws={'alpha':0.4, 'color':'#33C1FF'}, line_kws={'color':'black'})
axes[1].set_title('AudioLoudness vs. BeatsPerMinute', fontsize=15)

plt.suptitle('Relationship with Target Variable', fontsize=20)
plt.show()


# Energy and AudioLoudness were the most correlated features with the target.
combined_df['Energy_Loudness_Interact'] = combined_df['Energy'] * combined_df['AudioLoudness']

# Log transform on skeweed feature
# skewed_features = ['AcousticQuality', 'InstrumentalScore', 'VocalContent', 'LivePerformanceLikelihood']
# for feature in skewed_features:
#     combined_df[f'{feature}_log'] = np.log1p(combined_df[feature])

print("New features created. New shape of combined_df:", combined_df.shape)


from sklearn.preprocessing import StandardScaler
# Data scaling because our features have vastly different ranges.

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the data
scaled_df = pd.DataFrame(scaler.fit_transform(combined_df), columns=combined_df.columns)

print("Data has been scaled.")
scaled_df.head()


# Split the processed data back into training and testing sets
train_processed = scaled_df.iloc[:n_train]
test_processed = scaled_df.iloc[n_train:]

print("Feature engineering and scaling complete.")
print(f"New shape of training data: {train_processed.shape}")
print(f"New shape of test data: {test_processed.shape}")

# This function defines a single trial for the Optuna hyperparameter search.
def objective(trial):
    # --- Define Hyperparameter Search Space ---
    params = {
        'objective': 'regression_l1',
        'metric': 'rmse',
        'n_estimators': 2000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }

    # --- Run Cross-Validation to evaluate parameters ---
    NFOLDS = 5
    folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)
    oof_preds = np.zeros(train_processed.shape[0])

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_processed, target)):
        X_train, y_train = train_processed.iloc[train_idx], target.iloc[train_idx]
        X_valid, y_valid = train_processed.iloc[valid_idx], target.iloc[valid_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_valid, y_valid)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])

        oof_preds[valid_idx] = model.predict(X_valid)
        gc.collect()

    # Return the final RMSE score for this set of parameters
    rmse = np.sqrt(mean_squared_error(target, oof_preds))
    return rmse

# --- Create and Run the Optuna Study ---
study = optuna.create_study(direction='minimize')
print("Starting hyperparameter optimization... this may take some time.")
study.optimize(objective, n_trials=50) # You can adjust n_trials

print("\nOptimization finished.")
print(f"Best trial CV RMSE: {study.best_value}")
print("Best parameters found:")
print(study.best_params)


# Create & Run Optuna Study
study = optuna.create_study(direction='minimize')

print("Starting optimization... this may take some time.")
study.optimize(objective, n_trials=50)

# Analyze results
print("\nOptimization finished.")
print(f"Best trial CV RMSE: {study.best_value}")
print("Best parameters found:")
# Store the best parameters in a variable
best_params = study.best_params
print(best_params)

# Add fixed parameters needed for the final model
best_params['objective'] = 'regression_l1'
best_params['metric'] = 'rmse'
best_params['n_estimators'] = 2000
best_params['verbose'] = -1
best_params['n_jobs'] = -1
best_params['seed'] = 42

print("\nRe-training final model with best parameters...")


# Retrieve the best parameters found by Optuna
best_params = study.best_params

# Add fixed parameters needed for the final model
best_params['objective'] = 'regression_l1'
best_params['metric'] = 'rmse'
best_params['n_estimators'] = 2000
best_params['verbose'] = -1
best_params['n_jobs'] = -1
best_params['seed'] = 42

# --- K-Fold Training for the Final Model ---
print("\nRe-training final model with best parameters...")
NFOLDS = 5
folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)

# Initialize arrays for predictions
oof_preds_final = np.zeros(train_processed.shape[0])
sub_preds = np.zeros(test_processed.shape[0])

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_processed, target)):
    X_train, y_train = train_processed.iloc[train_idx], target.iloc[train_idx]
    X_valid, y_valid = train_processed.iloc[valid_idx], target.iloc[valid_idx]

    # Define the model with the best parameters
    model = lgb.LGBMRegressor(**best_params)
    
    # Train the model
    model.fit(X_train, y_train,
              eval_set=[(X_valid, y_valid)],
              callbacks=[lgb.early_stopping(100, verbose=False)])

    # Store validation predictions to calculate the final CV score
    oof_preds_final[valid_idx] = model.predict(X_valid)
    
    # Add test predictions (averaged over all folds)
    sub_preds += model.predict(test_processed) / folds.n_splits

    print(f"Fold {n_fold + 1} of final model re-training complete.")
    gc.collect()

# --- Evaluate Final CV Score ---
final_cv_rmse = np.sqrt(mean_squared_error(target, oof_preds_final))
print(f"\nTraining complete.")
print(f"Full CV RMSE Score for the final model: {final_cv_rmse}")


# --- Create Final Submission File ---
print("\nCreating final submission file...")
submission = pd.DataFrame({'id': test_ids, 'BeatsPerMinute': sub_preds})
submission.to_csv('submission_tuned.csv', index=False)

print("Tuned submission file 'submission_tuned.csv' created successfully!")
display(submission.head())

