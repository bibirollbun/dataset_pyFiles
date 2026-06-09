# import kagglehub
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.model_selection import StratifiedKFold
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from lightgbm import LGBMClassifier
# from sklearn.metrics import roc_auc_score

# # --- Load Competition Data ---
# COMP_PATH = "/kaggle/input/playground-series-s5e8/"
# train_synth = pd.read_csv(f"{COMP_PATH}train.csv")
# test_synth = pd.read_csv(f"{COMP_PATH}test.csv")
# sample_submission = pd.read_csv(f"{COMP_PATH}sample_submission.csv")

# # --- Download and Load Original Data ---
# # This downloads the dataset and returns the local path to it
# original_data_path = kagglehub.dataset_download("sushant097/bank-marketing-dataset-full")
# # Note: The original dataset often uses a semicolon (;) as a separator
# train_orig = pd.read_csv(f"{original_data_path}/bank-full.csv", sep=';')

# print("Data Loading Complete!")
# print(f"Synthetic Train Shape: {train_synth.shape}")
# print(f"Original Train Shape: {train_orig.shape}")
# print(f"Test Shape: {test_synth.shape}")


# # --- Harmonize Columns ---
# # The original dataset's target is 'y', let's rename it to match the competition's target.
# # The competition data uses 'y' as the target, but it's not a feature in train_orig yet.
# # The original target column is 'y', let's map 'yes'/'no' to 1/0.
# train_orig['y'] = train_orig['y'].map({'yes': 1, 'no': 0})

# # It looks like the column names are mostly consistent, but let's be safe.
# # We'll drop the 'id' from the synthetic training set for the combination.
# train_synth_no_id = train_synth.drop('id', axis=1)

# # --- Combine Datasets ---
# # Concatenate the original and synthetic training data
# df_train = pd.concat([train_synth_no_id, train_orig], ignore_index=True)
# print(f"Combined Training Data Shape: {df_train.shape}")


# # --- Quick EDA: Target Distribution ---
# plt.figure(figsize=(8, 5))
# sns.countplot(x='y', data=df_train, palette='viridis')
# plt.title('Target Distribution in Combined Data')
# plt.show()


# # --- Quick EDA: Age Distribution Comparison ---
# plt.figure(figsize=(12, 6))
# sns.histplot(train_synth['age'], color='skyblue', label='Synthetic', kde=True, stat="density", linewidth=0)
# sns.histplot(train_orig['age'], color='red', label='Original', kde=True, stat="density", linewidth=0)
# plt.title('Comparison of Age Distribution')
# plt.legend()
# plt.show()


# # --- Identify Feature Types ---
# # We exclude the target 'y' from the feature list
# features = [col for col in df_train.columns if col != 'y']
# categorical_features = df_train[features].select_dtypes(include='object').columns
# numerical_features = df_train[features].select_dtypes(include=['int64', 'float64']).columns

# # --- Create the Preprocessing Pipeline ---
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', StandardScaler(), numerical_features),
#         ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
#     ],
#     remainder='passthrough' # Keep other columns if any (shouldn't be the case here)
# )

# print("Preprocessing pipeline is ready.")


# # Define features (X) and target (y) from the combined dataframe
# X = df_train[features]
# y = df_train['y']

# # --- Cross-Validation Setup ---
# N_SPLITS = 5
# skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# oof_preds = np.zeros(len(df_train))
# test_preds = []

# # --- Training Loop ---
# for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
#     print(f"===== FOLD {fold+1} =====")
#     X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
#     X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

#     # Define the model within the pipeline
#     model = Pipeline(steps=[
#         ('preprocessor', preprocessor),
#         ('classifier', LGBMClassifier(random_state=42, is_unbalance=True))
#     ])

#     # Train the model
#     model.fit(X_train, y_train)

#     # Predict on validation set
#     val_preds = model.predict_proba(X_val)[:, 1]
#     oof_preds[val_idx] = val_preds

#     # Predict on test set (we'll average these later)
#     # The test set has an 'id' column which needs to be dropped before predicting
#     X_test = test_synth.drop('id', axis=1)
#     current_test_preds = model.predict_proba(X_test)[:, 1]
#     test_preds.append(current_test_preds)

#     # Evaluate fold
#     fold_auc = roc_auc_score(y_val, val_preds)
#     print(f"Fold {fold+1} AUC: {fold_auc}")

# # --- Overall CV Score ---
# overall_auc = roc_auc_score(y, oof_preds)
# print(f"\nOverall CV AUC: {overall_auc}")


# # Average the predictions from all folds for the test set
# avg_test_preds = np.mean(test_preds, axis=0)

# # --- Create Submission DataFrame ---
# submission_df = pd.DataFrame({'id': test_synth['id'], 'y': avg_test_preds})

# # --- Save to CSV ---
# submission_df.to_csv('submission.csv', index=False)

# print("\nSubmission file created successfully!")
# print(submission_df.head())


# Cell 1: Imports and Data Loading
import pandas as pd
import numpy as np
import kagglehub
import lightgbm as lgb
import xgboost as xgb
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

print("Step 1: Loading data...")

# --- Load Competition Data ---
COMP_PATH = "/kaggle/input/playground-series-s5e8/"
train_synth = pd.read_csv(f"{COMP_PATH}train.csv")
test_synth = pd.read_csv(f"{COMP_PATH}test.csv")

# --- Download and Load Original Data ---
original_data_path = kagglehub.dataset_download("sushant097/bank-marketing-dataset-full")
train_orig = pd.read_csv(f"{original_data_path}/bank-full.csv", sep=';')

# --- Harmonize and Combine Datasets ---
train_orig['y'] = train_orig['y'].map({'yes': 1, 'no': 0})
df_train = pd.concat([train_synth.drop('id', axis=1), train_orig], ignore_index=True)

print("Data loading and combination complete.")


# Cell 2: Feature Engineering
print("\nStep 2: Engineering new features...")

all_dfs = [df_train, test_synth]

for df in all_dfs:
    df['balance_per_age'] = df['balance'] / (df['age'] + 1e-6)
    df['duration_per_campaign'] = df['duration'] / (df['campaign'] + 1e-6)
    df['age_squared'] = df['age']**2

print("New features created successfully.")


# Cell 3: Preprocessing
print("\nStep 3: Defining variables and the preprocessing pipeline...")

features = [col for col in df_train.columns if col != 'y']
X = df_train[features]
y = df_train['y']

categorical_features = X.select_dtypes(include='object').columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

print("Preprocessing pipeline is ready.")


# Cell 4: FAST Hyperparameter Tuning
# print("\nStep 4: Finding best LightGBM parameters with Optuna (FAST and GPU)...")

# def objective(trial):
#     params = {
#         'objective': 'binary', 'metric': 'auc', 'random_state': 42, 'is_unbalance': True,
#         'device': 'gpu',  # Use the GPU
#         'n_estimators': trial.suggest_int('n_estimators', 400, 1500),
#         'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.2),
#         'num_leaves': trial.suggest_int('num_leaves', 20, 150),
#         'max_depth': trial.suggest_int('max_depth', 5, 10),
#         'lambda_l1': trial.suggest_float('lambda_l1', 1e-6, 10.0, log=True),
#         'lambda_l2': trial.suggest_float('lambda_l2', 1e-6, 10.0, log=True),
#     }

#     model = lgb.LGBMClassifier(**params)
#     pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
#     skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) # 3 splits for speed
#     score = cross_val_score(pipeline, X, y, cv=skf, scoring='roc_auc')
#     return score.mean()

# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=20) # 20 trials for speed

# best_lgbm_params = study.best_params
# print(f"\nBest LGBM AUC from study: {study.best_value}")
# print("Best LGBM parameters found: ", best_lgbm_params)


# Cell 5: Final Training and Ensembling
print("\nStep 5: Training final models and ensembling...")
best_lgbm_params1 =  {'n_estimators': 1149, 'learning_rate': 0.05791355852287643, 'num_leaves': 56, 'max_depth': 7, 'lambda_l1': 9.492534920573037, 'lambda_l2': 2.921457674144133}
# --- 1. LightGBM with best parameters and GPU ---
final_lgbm_params = best_lgbm_params1 #best_lgbm_params 
final_lgbm_params['device'] = 'gpu'
final_lgbm_params['is_unbalance'] = True
final_lgbm_params['random_state'] = 42

final_lgbm = lgb.LGBMClassifier(**final_lgbm_params)
lgbm_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', final_lgbm)])
lgbm_pipeline.fit(X, y)
preds_lgbm = lgbm_pipeline.predict_proba(test_synth[features])[:, 1]
print("Final LGBM model trained.")

# --- 2. XGBoost model with GPU ---
scale_pos_weight = y.value_counts()[0] / y.value_counts()[1]
final_xgb = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    tree_method='gpu_hist'  # Enable GPU for XGBoost
)
xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', final_xgb)])
xgb_pipeline.fit(X, y)
preds_xgb = xgb_pipeline.predict_proba(test_synth[features])[:, 1]
print("Final XGBoost model trained.")

# --- 3. Ensemble (combine) predictions ---
final_preds = 0.6 * preds_lgbm + 0.4 * preds_xgb
print("Predictions ensembled.")


# Cell 6: Create Submission File
print("\nStep 6: Creating submission file...")

submission_df = pd.DataFrame({'id': test_synth['id'], 'y': final_preds})
submission_df.to_csv('submission.csv', index=False)

print("Optimized submission file created successfully!")
print(submission_df.head())

