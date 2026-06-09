

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load the datasets
df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
dft = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

print("Training data loaded. Shape:", df.shape)
print("Testing data loaded. Shape:", dft.shape)


# Get a concise summary of the dataframe
print("--- Data Info ---")
df.info()

print("\n\n--- Statistical Summary ---")
print(df.describe())

print("\n\n--- Missing Values per Column ---")
print(df.isnull().sum())


df


df.describe()


import seaborn as sns
import matplotlib.pyplot as plt
continuous_features = ['Time_spent_Alone', 'Going_outside', 'Drained_after_socializing', 'Post_frequency']
discrete_features = ['Stage_fear', 'Social_event_attendance', 'Friends_circle_size']
for col in continuous_features + discrete_features:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=['Personality'])

# KDE Plots (Continuous)
for col in continuous_features:
    plot_df = df[[col, 'Personality']].dropna()
    if plot_df.empty:
        print(f"Skipping KDE plot for {col} (no valid data).")
        continue

    plt.figure(figsize=(8, 4))
    sns.kdeplot(data=plot_df, x=col, hue='Personality', fill=True, common_norm=False, bw_adjust=0.5)
    plt.title(f"KDE Plot of {col} by Personality")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.legend(title='Personality')
    plt.tight_layout()
    plt.show()

# Count Plots (Discrete)
for col in discrete_features:
    plot_df = df[[col, 'Personality']].dropna()
    if plot_df.empty:
        print(f"Skipping bar plot for {col} (no valid data).")
        continue

    plt.figure(figsize=(8, 4))
    sns.countplot(data=plot_df, x=col, hue='Personality')
    plt.title(f"Bar Plot of {col} by Personality")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.legend(title='Personality')
    plt.tight_layout()
    plt.show()



df=pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
ordinal_map = {
    'No': 0,
    'Yes': 1
}
ordinal_map_target = {
    'Introvert': 0,
    'Extrovert': 1
}

df['Stage_fear'] = df['Stage_fear'].map(ordinal_map)
df['Drained_after_socializing'] = df['Drained_after_socializing'].map(ordinal_map)
df['Personality'] = df['Personality'].map(ordinal_map_target)



df.info()


df.isnull().sum()



import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Keep only numeric columns (correlation only works with numbers)
numeric_df = df.select_dtypes(include='number')

# Step 2: Compute correlation matrix
corr_matrix = numeric_df.corr()

# Step 3: Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.tight_layout()
plt.show()



corr_matrix 


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Count number of missing columns per row
missing_per_row = df.isnull().sum(axis=1)

# Step 2: Count how many rows have 0, 1, 2, ... missing columns
missing_counts = missing_per_row.value_counts().sort_index()

# Step 3: Plot
plt.figure(figsize=(8, 5))
sns.barplot(x=missing_counts.index, y=missing_counts.values, palette='viridis')

plt.title('Distribution of Missing Values per Row')
plt.xlabel('Number of Missing Features in Row')
plt.ylabel('Number of Rows')
plt.xticks(missing_counts.index)  # Make sure all bars are labeled
plt.tight_layout()
plt.show()



dft=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
dft


# le_dict = {} 
# df=pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
# dft=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")

# import pandas as pd
# import numpy as np
# from xgboost import XGBRegressor, XGBClassifier
# from sklearn.metrics import mean_absolute_error, accuracy_score
# from sklearn.preprocessing import LabelEncoder
# from tqdm import tqdm

# # Load your data (assuming df and dft already exist)
# df_full = df.copy()
# dft_full = dft.copy()

# # --- SETTINGS ---
# label_cols = ['Stage_fear', 'Drained_after_socializing']
# features = df_full.columns.drop(['id', 'Personality'])

# from sklearn.preprocessing import LabelEncoder

# def safe_label_encode(series):
#     le = LabelEncoder()
#     non_null = series.dropna().astype(str)
#     le.fit(non_null)
#     return le

# # Then apply it like this:
# for col in label_cols:
#     le = safe_label_encode(df_full[col])
#     le_dict[col] = le
#     df_full[col] = df_full[col].map(lambda x: le.transform([str(x)])[0] if pd.notna(x) else np.nan)
#     dft_full[col] = dft_full[col].map(lambda x: le.transform([str(x)])[0] if pd.notna(x) else np.nan)


# # --- STEP 2: Filter df rows with â‰¤1 missing value ---
# train_features_only = df_full[features]
# null_counts = train_features_only.isnull().sum(axis=1)
# df_cleaned = df_full[null_counts <= 1].copy().reset_index(drop=True)

# # --- STEP 3: 100-row validation set ---
# val_set = df_cleaned.iloc[:100].copy()
# df_train = df_cleaned.iloc[100:].copy()

# # --- STEP 4: Train XGBoost Models ---
# def is_classification(feature):
#     return feature in label_cols

# models = {}

# def train_and_evaluate_feature_model(feature, df_train, val_set):
#     input_features = [f for f in features if f != feature]
#     train_subset = df_train.dropna(subset=[feature] + input_features)
#     val_subset = val_set.dropna(subset=[feature] + input_features)

#     X_train = train_subset[input_features]
#     y_train = train_subset[feature]
#     X_val = val_subset[input_features]
#     y_val = val_subset[feature]

#     if is_classification(feature):
#         model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', verbosity=0)
#     else:
#         model = XGBRegressor(n_estimators=100, verbosity=0)

#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_val)

#     if is_classification(feature):
#         score = accuracy_score(y_val, y_pred)
#         print(f"âœ… {feature} - Accuracy: {score:.4f}")
#     else:
#         score = mean_absolute_error(y_val, y_pred)
#         print(f"âœ… {feature} - MAE: {score:.4f}")

#     print(pd.DataFrame({'True': y_val.values[:5], 'Predicted': y_pred[:5]}))
#     return model

# print("ğŸ”„ Training models...")
# for feature in tqdm(features, desc="Training"):
#     model = train_and_evaluate_feature_model(feature, df_train, val_set)
#     models[feature] = model

# # --- STEP 5: Imputation Logic ---
# def impute_single_missing(df_in, features, models):
#     df = df_in.copy()
#     row_null_counts = df[features].isnull().sum(axis=1)
#     one_missing_rows = df[row_null_counts == 1]
#     for idx, row in one_missing_rows.iterrows():
#         missing_col = row[features].isnull().idxmax()
#         input_cols = [f for f in features if f != missing_col]
#         if row[input_cols].isnull().any():  # skip if other features are also NaN
#             continue
#         model = models.get(missing_col)
#         if model:
#             inp = row[input_cols].values.reshape(1, -1)
#             pred = model.predict(inp)[0]
#             df.at[idx, missing_col] = pred
#     return df

# def fill_multi_missing_with_mean(df, features, ref_df):
#     df = df.copy()
#     for col in features:
#         mean_val = ref_df[col].mean()
#         df[col] = df[col].fillna(mean_val)
#     return df

# # --- STEP 6: Final Cleanup ---
# # For df: drop rows with >1 missing, then impute 1-missing rows
# null_counts_df = df_full[features].isnull().sum(axis=1)
# df_dropped = df_full[null_counts_df > 1].copy()
# df_clean_part = df_full[null_counts_df <= 1].copy()
# df_clean_part = impute_single_missing(df_clean_part, features, models)
# df_final = df_clean_part.dropna(subset=features).copy().reset_index(drop=True)  # Final training set

# # For dft: split into 1-missing and >1-missing, process accordingly
# null_counts_dft = dft_full[features].isnull().sum(axis=1)
# dft_one_missing = dft_full[null_counts_dft == 1].copy()
# dft_multi_missing = dft_full[null_counts_dft > 1].copy()
# dft_none_missing = dft_full[null_counts_dft == 0].copy()

# # Impute 1-missing
# dft_one_filled = impute_single_missing(dft_one_missing, features, models)

# # Fill >1 missing with mean
# dft_multi_filled = fill_multi_missing_with_mean(dft_multi_missing, features, df_final)

# # Combine
# dft_final = pd.concat([dft_none_missing, dft_one_filled, dft_multi_filled], ignore_index=True)
# dft_final = dft_final.sort_values(by='id').reset_index(drop=True)



import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
import lightgbm as lgb # Import the lightgbm library
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import optuna
from collections import Counter
import warnings

# Suppress Optuna's trial info messages and other warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


print("--- Step 1: Loading Data ---")

df=pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
dft=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")

df_full = df.copy()
dft_full = dft.copy()

print(f"Train data loaded with shape: {df.shape}")
print(f"Test data loaded with shape: {dft.shape}")

print("\n--- Step 2: Preprocessing and Data Splitting ---")
# --- SETTINGS ---
le_dict = {}
label_cols = ['Stage_fear', 'Drained_after_socializing']
features = [col for col in df_full.columns if col not in ['id', 'Personality']]

# --- Label Encode Categorical Features ---
for col in label_cols:
    le = LabelEncoder()
    # Fit on non-null values from both train and test sets to capture all possible categories
    all_values = pd.concat([df_full[col].dropna(), dft_full[col].dropna()]).astype(str).unique()
    le.fit(all_values)
    le_dict[col] = le
    # Transform train set
    df_full[col] = df_full[col].map(lambda x: le.transform([str(x)])[0] if pd.notna(x) else np.nan)
    # Transform test set
    dft_full[col] = dft_full[col].map(lambda x: le.transform([str(x)])[0] if pd.notna(x) else np.nan)

# --- Create clean training and validation sets for imputation models ---
train_features_only = df_full[features]
null_counts = train_features_only.isnull().sum(axis=1)
df_cleaned = df_full[null_counts <= 1].copy().reset_index(drop=True)

# Use a small part of the clean data for validation during hyperparameter tuning
val_set = df_cleaned.iloc[:500].copy() # Using a larger validation set for robustness
df_train = df_cleaned.iloc[500:].copy()

print(f"Cleaned data for training imputation models: {df_train.shape}")
print(f"Validation set for tuning: {val_set.shape}")


def is_classification(feature):
    return feature in label_cols

print("\n--- Step 3: Hyperparameter Tuning and Model Training ---")

def find_best_params_and_train(feature, X_train, y_train, X_val, y_val):
    """
    Finds best parameters using Optuna and trains three models (XGB, LGBM, CatBoost).
    Returns a dictionary of trained models.
    """
    task_type = 'classifier' if is_classification(feature) else 'regressor'
    trained_models = {}

    for model_type in ['xgb', 'lgbm', 'catboost']:

        def objective(trial):
            # Define model and parameters based on model_type
            if model_type == 'xgb':
                param = {
                    'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 8),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
                    'verbosity': 0,
                }
                if task_type == 'classifier':
                    model = XGBClassifier(**param, use_label_encoder=False, eval_metric='logloss')
                else:
                    model = XGBRegressor(**param, eval_metric='mae')

            elif model_type == 'lgbm':
                param = {
                    'objective': 'binary' if task_type == 'classifier' else 'regression_l1',
                    'metric': 'binary_logloss' if task_type == 'classifier' else 'mae',
                    'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 8),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
                    'verbose': -1
                }
                if task_type == 'classifier':
                    model = lgb.LGBMClassifier(**param)
                else:
                    model = lgb.LGBMRegressor(**param)
            
            else: # catboost
                param = {
                    'iterations': trial.suggest_int('iterations', 200, 1000),
                    'depth': trial.suggest_int('depth', 3, 8),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
                    'verbose': 0
                }
                if task_type == 'classifier':
                    model = CatBoostClassifier(**param)
                else:
                    model = CatBoostRegressor(**param)

            # --- MODIFIED FIT CALL ---
            if model_type == 'lgbm':
                # LGBM uses a callback for early stopping
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                          callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])
            else:
                # XGBoost and CatBoost can take the argument directly
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=30, verbose=False)
            
            preds = model.predict(X_val)

            if task_type == 'classifier':
                return accuracy_score(y_val, preds)
            else:
                return mean_absolute_error(y_val, preds)

        study_direction = 'maximize' if task_type == 'classifier' else 'minimize'
        study = optuna.create_study(direction=study_direction)
        study.optimize(objective, n_trials=15) # n_trials can be increased for better results

        best_params = study.best_params
        
        # Instantiate and train the final model with the best parameters
        if model_type == 'xgb':
            final_model = XGBClassifier(**best_params, use_label_encoder=False, verbosity=0) if task_type == 'classifier' else XGBRegressor(**best_params, verbosity=0)
        elif model_type == 'lgbm':
            # Add objective and metric for LGBM final model
            best_params['objective'] = 'binary' if task_type == 'classifier' else 'regression_l1'
            best_params['metric'] = 'binary_logloss' if task_type == 'classifier' else 'mae'
            final_model = lgb.LGBMClassifier(**best_params, verbose=-1) if task_type == 'classifier' else lgb.LGBMRegressor(**best_params, verbose=-1)
        else: # catboost
            final_model = CatBoostClassifier(**best_params, verbose=0) if task_type == 'classifier' else CatBoostRegressor(**best_params, verbose=0)
        
        final_model.fit(X_train, y_train)
        trained_models[model_type] = final_model
        
    return trained_models

# --- Train all models for all features ---
all_models = {}
for feature in tqdm(features, desc="Training Imputation Models"):
    input_features = [f for f in features if f != feature]
    
    train_subset = df_train.dropna(subset=[feature] + input_features)
    val_subset = val_set.dropna(subset=[feature] + input_features)
    
    X_train = train_subset[input_features]
    y_train = train_subset[feature]
    X_val = val_subset[input_features]
    y_val = val_subset[feature]

    if len(X_train) > 100 and len(X_val) > 10:
        all_models[feature] = find_best_params_and_train(feature, X_train, y_train, X_val, y_val)


print("\n--- Step 4: Ensemble Imputation ---")
def impute_single_missing_ensemble(df_in, features, all_models):
    df = df_in.copy()
    row_null_counts = df[features].isnull().sum(axis=1)
    one_missing_rows = df[row_null_counts == 1]
    
    for idx, row in tqdm(one_missing_rows.iterrows(), total=len(one_missing_rows), desc="Imputing 1-missing"):
        missing_col = row[features].isnull().idxmax()
        
        if missing_col in all_models:
            input_cols = [f for f in features if f != missing_col]
            models = all_models[missing_col]
            
            if row[input_cols].isnull().any():
                continue
                
            inp = row[input_cols].values.reshape(1, -1)
            preds = [model.predict(inp)[0] for model in models.values()]

            if is_classification(missing_col):
                ensemble_pred = Counter(preds).most_common(1)[0][0]
            else:
                ensemble_pred = np.mean(preds)
            
            df.at[idx, missing_col] = ensemble_pred
            
    return df

def fill_remaining_with_central_tendency(df, features, ref_df):
    df_out = df.copy()
    for col in tqdm(features, desc="Imputing multi-missing"):
        if is_classification(col):
            fill_value = ref_df[col].mode()[0]
        else:
            fill_value = ref_df[col].mean()
        df_out[col].fillna(fill_value, inplace=True)
    return df_out

print("\n--- Step 5: Applying Imputation to Final Datasets ---")

# --- Process Training Data ---
null_counts_df = df_full[features].isnull().sum(axis=1)
df_none_missing = df_full[null_counts_df == 0].copy()
df_one_missing = df_full[null_counts_df == 1].copy()
df_multi_missing = df_full[null_counts_df > 1].copy()

df_one_filled = impute_single_missing_ensemble(df_one_missing, features, all_models)
df_ref = pd.concat([df_none_missing, df_one_filled]).dropna(subset=features)
df_multi_filled = fill_remaining_with_central_tendency(df_multi_missing, features, df_ref)
df_final = pd.concat([df_none_missing, df_one_filled, df_multi_filled], ignore_index=True).sort_values(by='id').reset_index(drop=True)
df_final = df_final.dropna(subset=features)

# --- Process Test Data ---
null_counts_dft = dft_full[features].isnull().sum(axis=1)
dft_none_missing = dft_full[null_counts_dft == 0].copy()
dft_one_missing = dft_full[null_counts_dft == 1].copy()
dft_multi_missing = dft_full[null_counts_dft > 1].copy()

dft_one_filled = impute_single_missing_ensemble(dft_one_missing, features, all_models)
dft_multi_filled = fill_remaining_with_central_tendency(dft_multi_missing, features, df_final)
dft_final = pd.concat([dft_none_missing, dft_one_filled, dft_multi_filled], ignore_index=True).sort_values(by='id').reset_index(drop=True)


print("\n--- Step 6: Final Results ---")
print(f"Original train shape: {df.shape}")
print(f"Final imputed train shape: {df_final.shape}")
print(f"Train NaNs remaining: {df_final[features].isnull().sum().sum()}")

print(f"\nOriginal test shape: {dft.shape}")
print(f"Final imputed test shape: {dft_final.shape}")
print(f"Test NaNs remaining: {dft_final[features].isnull().sum().sum()}")

print("\nImputation process complete.")

print("\nFinal Training DataFrame Head:")
print(df_final.head())
print("\nFinal Test DataFrame Head:")
print(dft_final.head())


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# --- 1. Prepare data ---
X = df_final.drop(columns=['id', 'Personality'])
y = df_final['Personality']

# Encode target labels: "Introvert" â†’ 0, "Extrovert" â†’ 1
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

# --- 2. Split into train/val sets ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# --- 3. Define Base Models ---
lgbm_params = {
    'objective': 'binary',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'random_state': 42,
    'num_leaves': 259,
    'learning_rate': 0.029385442161128397,
    'n_estimators': 369,
    'subsample_for_bin': 175904,
    'reg_alpha': 2.2990464275110867,
    'reg_lambda': 0.009126367790369154,
    'max_depth': 8,
    'colsample_bytree': 0.368023545639538,
    'subsample': 0.7112554200243772,
    'min_child_samples': 100,
    'feature_fraction': 0.570042007618262,
    'bagging_fraction': 0.7591648261818684
}

lgbm = LGBMClassifier(**lgbm_params)

xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

cat = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    verbose=0,
    random_state=42
)

# --- 4. Define Meta Learner ---
meta_learner = LogisticRegression(max_iter=1000)

# --- 5. Build the Stacking Ensemble ---
stacked_model = StackingClassifier(
    estimators=[
        ('lgbm', lgbm),
        ('xgb', xgb),
        ('cat', cat)
    ],
    final_estimator=meta_learner,
    cv=5,           # 5-fold cross-validation on base models
    n_jobs=-1,
    passthrough=False
)

# --- 6. Train the Ensemble ---
stacked_model.fit(X_train, y_train)

# --- 7. Evaluate on validation set ---
y_val_pred = stacked_model.predict(X_val)
print("ğŸ“Š Classification Report (Validation Set):")
print(classification_report(y_val, y_val_pred, target_names=target_encoder.classes_))

# --- 8. Predict on dft_final ---
X_test = dft_final.drop(columns=['id'])
y_test_pred = stacked_model.predict(X_test)
y_test_labels = target_encoder.inverse_transform(y_test_pred)

# --- 9. Save predictions back to test dataframe ---
dft_final_with_preds = dft_final.copy()
dft_final_with_preds['Predicted_Personality'] = y_test_labels



submission = dft_final_with_preds[['id', 'Predicted_Personality']].copy()
submission.columns = ['id', 'Personality']  # Rename to match submission format

# Save to CSV
submission.to_csv("submission.csv", index=False)

print("submission.csv created successfully!")


# import xgboost as xgb
# import optuna
# from tqdm.notebook import tqdm
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# # --- 1. Prepare the data (Assuming df_final is your training dataframe) ---
# # This part is necessary to create a validation set for tuning
# X = df_final.drop(columns=['id', 'Personality'])
# y = df_final['Personality']

# # Encode target labels
# target_encoder = LabelEncoder()
# y_encoded = target_encoder.fit_transform(y)

# # Split data to create a validation set for evaluating parameters
# X_train, X_val, y_train, y_val = train_test_split(
#     X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
# )


# # --- 2. Define the Objective Function for Optuna ---
# # Optuna will try to find the parameters that maximize the return value (accuracy)
# def objective(trial):
#     """Defines the parameter search space and model training for one trial."""
#     params = {
#         'objective': 'binary:logistic',
#         'eval_metric': 'logloss',
#         'use_label_encoder': False,
#         'tree_method': 'gpu_hist',        # <---- Use GPU
#         'predictor': 'gpu_predictor',     # <---- Use GPU
        
#         # --- Hyperparameters to Tune ---
#         'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#         'max_depth': trial.suggest_int('max_depth', 4, 12),
#         'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#         'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
#         'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
#         'random_state': 42
#     }

#     model = xgb.XGBClassifier(**params)
    
#     # Train the model with early stopping to prune unpromising trials
#     model.fit(
#         X_train, y_train,
#         eval_set=[(X_val, y_val)],
#         early_stopping_rounds=50,
#         verbose=False
#     )

#     # Evaluate the model on the validation set
#     y_pred = model.predict(X_val)
#     accuracy = accuracy_score(y_val, y_pred)
    
#     return accuracy


# # --- 3. Run the Hyperparameter Tuning with a Progress Bar ---
# N_TRIALS = 100  # Number of different parameter combinations to test

# # Create a study object to manage optimization. We want to maximize accuracy.
# study = optuna.create_study(direction='maximize')

# # Create a tqdm progress bar
# pbar = tqdm(total=N_TRIALS, desc="Tuning Hyperparameters")

# # Run the optimization and update the progress bar with a callback
# study.optimize(
#     objective, 
#     n_trials=N_TRIALS, 
#     callbacks=[lambda study, trial: pbar.update()]
# )

# pbar.close()


# # --- 4. Print the Best Parameters Found ---
# print("\n" + "="*50)
# print("âœ… Tuning Complete!")
# print(f"ğŸ�† Best Validation Accuracy: {study.best_value:.4f}")
# print("ğŸ“‹ Best Hyperparameters Found:")

# # Print parameters in a clean, readable format
# for key, value in study.best_params.items():
#     print(f"    '{key}': {value}")
# print("="*50)




