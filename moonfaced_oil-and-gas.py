import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb
import catboost as cb # <--- Import CatBoost
import warnings
import os
import traceback

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Configuration ---
TRAIN_FILE = '/kaggle/input/oilgas-field-prediction/train_oil.csv'
TEST_FILE = '/kaggle/input/oilgas-field-prediction/oil_test.csv'
SUBMISSION_FILE = 'submission_catboost_tuned.csv' # <--- New submission file name
TARGET_COL = 'Onshore/Offshore' # Verified from previous run
N_CV_SPLITS = 5
N_TUNING_ITER = 60 # <--- Increased iterations slightly for wider search

# --- Global Variables ---
X = None
y = None
X_test = None
label_encoder = None
test_indices = None
scale_pos_weight_val = 1.0

# --- 1. Load Data & Validate Target ---
print(f"--- Loading Data ---")
# (Keep the robust loading and target validation logic from the previous version)
# ... (Assume loading and target validation code is here and works) ...
try:
    train_df_raw = pd.read_csv(TRAIN_FILE)
    test_df_raw = pd.read_csv(TEST_FILE)

    if train_df_raw.columns[0].lower() in ['unnamed: 0', 'index']:
         train_df = pd.read_csv(TRAIN_FILE, index_col=0)
    else: train_df = train_df_raw.copy()
    if test_df_raw.columns[0].lower() in ['unnamed: 0', 'index']:
         test_df = pd.read_csv(TEST_FILE, index_col=0)
    else: test_df = test_df_raw.copy()

    test_indices = test_df.index
    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"\nColumns found in train_df: {train_df.columns.tolist()}")

    print("\n--- Validating Target Column ---")
    if TARGET_COL not in train_df.columns:
        print(f"Error: Target column '{TARGET_COL}' not found.")
        exit()
    print(f"Confirmed target column: '{TARGET_COL}'")

    print("\n--- Cleaning Target Variable ---")
    train_df[TARGET_COL] = train_df[TARGET_COL].astype(str).str.upper().replace('OﬀSHORE', 'OFFSHORE', regex=False)
    initial_rows = len(train_df)
    train_df.dropna(subset=[TARGET_COL], inplace=True)
    rows_after_na = len(train_df); print(f"Dropped {initial_rows - rows_after_na} rows with missing target values.") if initial_rows > rows_after_na else None
    train_df = train_df[train_df[TARGET_COL].isin(['ONSHORE', 'OFFSHORE'])]
    rows_after_filter = len(train_df); print(f"Dropped {rows_after_na - rows_after_filter} rows with invalid target values.") if rows_after_na > rows_after_filter else None
    if len(train_df) == 0: print("Error: No valid data remaining."); exit()

    print("Target variable distribution (Train - Cleaned):"); print(train_df[TARGET_COL].value_counts(dropna=False))

    X = train_df.drop(TARGET_COL, axis=1)
    y = train_df[TARGET_COL]
    print(f"Features (X) shape: {X.shape}"); print(f"Target (y) shape: {y.shape}")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print("\nTarget Encoding Mapping:")
    onshore_label = -1; offshore_label = -1
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"{class_name}: {i}");
        if class_name == 'ONSHORE': onshore_label = i
        if class_name == 'OFFSHORE': offshore_label = i
    neg_count = np.sum(y_encoded == offshore_label); pos_count = np.sum(y_encoded == onshore_label)
    scale_pos_weight_val = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"Calculated scale_pos_weight (for Onshore={onshore_label}): {scale_pos_weight_val:.2f}")

    original_cols = X.columns.tolist()
    cols_to_select_in_test = [col for col in original_cols if col in test_df.columns]
    if len(cols_to_select_in_test) < len(original_cols): print(f"\nWarning: Columns missing in test data: {[col for col in original_cols if col not in test_df.columns]}.")
    X_test = test_df[cols_to_select_in_test].copy()
    print(f"Initial Test Features (X_test) shape: {X_test.shape}")

except Exception as e:
    print(f"An error occurred during data loading or initial cleaning: {e}"); traceback.print_exc(); exit()

if X is None or X_test is None or y is None: print("Error: Data loading failed."); exit()

# --- 2. Feature Engineering ---
print("\n--- Feature Engineering ---")
# (Function remains the same)
def create_features(df):
    df_out = df.copy()
    gross_thick_col = 'Thickness (gross average ft)'
    net_thick_col = 'Thickness (net pay average ft)'
    if gross_thick_col in df_out.columns and net_thick_col in df_out.columns:
        gross_numeric = pd.to_numeric(df_out[gross_thick_col], errors='coerce').fillna(0)
        net_numeric = pd.to_numeric(df_out[net_thick_col], errors='coerce').fillna(0)
        epsilon = 1e-6
        df_out['Net_Pay_Ratio'] = net_numeric / (gross_numeric + epsilon)
        df_out['Net_Pay_Ratio'] = df_out['Net_Pay_Ratio'].clip(0, 1.5)
        df_out.loc[gross_numeric <= 0, 'Net_Pay_Ratio'] = 0
        print("Created 'Net_Pay_Ratio' feature.")
    else:
        print("Warning: Thickness columns needed for 'Net_Pay_Ratio' not found.")
    return df_out

X = create_features(X)
X_test = create_features(X_test)
print(f"Shape after feature engineering - Train: {X.shape}, Test: {X_test.shape}")


# --- 3. Identify Feature Types (Post-Engineering) ---
print("\n--- Identifying Feature Types ---")
# (Logic remains the same)
numeric_features = []
potential_num_cols = X.select_dtypes(include=np.number).columns.tolist()
potential_num_cols.extend(['Latitude', 'Longitude', 'Depth', 'Thickness (gross average ft)', 'Thickness (net pay average ft)', 'Porosity', 'Permeability']) # Adjust 'Depth' if name differs
potential_num_cols.append('Net_Pay_Ratio'); potential_num_cols = list(set(col for col in potential_num_cols if col in X.columns))
print("Attempting numeric conversion for columns:")
for col in potential_num_cols:
    try:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        if col in X_test.columns: X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
        if pd.api.types.is_numeric_dtype(X[col]) and X[col].notna().any(): numeric_features.append(col)
    except Exception as e: print(f"   - Error converting column '{col}': {e}")
numeric_features = list(set(numeric_features))
identifier_cols = ['Field name', 'Reservoir unit']; categorical_features = [col for col in X.columns if col not in numeric_features and col not in identifier_cols and col in X.columns]
print(f"\nIdentified Numerical Features ({len(numeric_features)}): {numeric_features}")
print(f"Identified Categorical Features ({len(categorical_features)}): {categorical_features}")
X = X.drop(columns=identifier_cols, errors='ignore'); X_test = X_test.drop(columns=identifier_cols, errors='ignore')
numeric_features = [f for f in numeric_features if f in X.columns]; categorical_features = [f for f in categorical_features if f in X.columns]


# --- 4. Preprocessing Pipeline ---
print("\n--- Creating Preprocessing Pipeline ---")
# (Pipeline definition remains the same - OHE for all models including CatBoost for simplicity now)
numerical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
categorical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='_MISSING_')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
preprocessor = ColumnTransformer(transformers=[('num', numerical_pipeline, numeric_features), ('cat', categorical_pipeline, categorical_features)], remainder='passthrough', n_jobs=-1)


# --- 5. Model Definition (Including CatBoost) ---
print("\n--- Defining Models ---")
# NOTE: CatBoost is sensitive to learning rate and iterations. Added verbose=0 to silence its output during CV/tuning.
# Added 'auto_class_weights':'Balanced' as an alternative to scale_pos_weight for CatBoost
models_to_evaluate = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced', solver='liblinear'),
    "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
    "XGBoost": xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight_val),
    "LightGBM": lgb.LGBMClassifier(random_state=42, class_weight='balanced', n_jobs=-1, verbosity=-1), # verbosity=-1 to silence
    "CatBoost": cb.CatBoostClassifier(random_state=42, eval_metric='Accuracy', scale_pos_weight=scale_pos_weight_val, verbose=0) # Use scale_pos_weight or auto_class_weights
    # "CatBoost_Balanced": cb.CatBoostClassifier(random_state=42, eval_metric='Accuracy', auto_class_weights='Balanced', verbose=0) # Alternative imbalance handling
}


# --- 6. Initial Model Comparison (Cross-Validation) ---
print("\n--- Initial Model Comparison (Cross-Validation) ---")
results = {}
cv = StratifiedKFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=42)
for name, model in models_to_evaluate.items():
    pipeline = Pipeline([('preprocess', preprocessor), ('classifier', model)])
    try:
        scores = cross_val_score(pipeline, X, y_encoded, cv=cv, scoring='accuracy', n_jobs=-1)
        results[name] = scores.mean()
        print(f"{name}: Mean Accuracy = {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    except Exception as e:
        print(f"Error evaluating {name}: {e}")
        results[name] = 0.0


# --- 7. Hyperparameter Tuning (Multiple Models) ---
print(f"\n--- Hyperparameter Tuning (Top Models) using RandomizedSearchCV ---")

# Define parameter grids (Add CatBoost, Adjust RF for less overfitting)
param_grids = {
    "Random Forest": {
        'classifier__n_estimators': [100, 200, 300, 400],
        'classifier__max_depth': [5, 10, 15, 20], # <-- Limit max_depth
        'classifier__min_samples_split': [5, 10, 15], # <-- Increase min samples
        'classifier__min_samples_leaf': [3, 5, 7],    # <-- Increase min samples
        'classifier__max_features': ['sqrt', 'log2'] # Keep effective options
    },
    "XGBoost": { # Keep previous grid, maybe add more regularization if needed
        'classifier__n_estimators': [100, 200, 300, 500], 'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7, 9], 'classifier__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'classifier__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0], 'classifier__gamma': [0, 0.1, 0.2, 0.3, 0.5],
        'classifier__reg_alpha': [0, 0.01, 0.1, 1], 'classifier__reg_lambda': [0, 0.1, 1, 5] # Added more lambda
    },
    "LightGBM": { # Keep previous grid, maybe add more regularization
        'classifier__n_estimators': [100, 200, 300, 500], 'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7, 9, -1], 'classifier__num_leaves': [15, 31, 63], # Limit num_leaves
        'classifier__subsample': [0.6, 0.7, 0.8, 0.9, 1.0], 'classifier__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'classifier__reg_alpha': [0, 0.01, 0.1, 1], 'classifier__reg_lambda': [0, 0.1, 1, 5] # Added more lambda
    },
    "CatBoost": {
        'classifier__iterations': [100, 200, 300, 500], # ~n_estimators
        'classifier__learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
        'classifier__depth': [4, 6, 8, 10], # ~max_depth
        'classifier__l2_leaf_reg': [1, 3, 5, 7, 9], # Lambda regularization
        'classifier__border_count': [32, 64, 128], # For numerical features
        'classifier__subsample': [0.6, 0.7, 0.8, 0.9, 1.0] # If supported and useful
        # 'classifier__colsample_bylevel': [0.6, 0.7, 0.8, 0.9, 1.0] # Alternative to colsample_bytree
    }
    # Add "CatBoost_Balanced" grid if using that model variation
}

# Select models to tune (e.g., top performers from initial CV or all gradient boosters + RF)
models_to_tune = ["Random Forest", "XGBoost", "LightGBM", "CatBoost"] # Adjust as needed
# models_to_tune = [name for name, score in sorted(results.items(), key=lambda item: item[1], reverse=True)[:3]] # Tune top 3


best_overall_score = -1
best_pipeline = None
best_tuned_model_name = ""

for name in models_to_tune:
    if name not in models_to_evaluate or name not in param_grids:
        print(f"Skipping tuning for {name} (not defined or no grid).")
        continue

    print(f"\n--- Tuning {name} ---")
    base_model = models_to_evaluate[name]
    param_dist = param_grids[name]

    tuning_pipeline = Pipeline([('preprocess', preprocessor), ('classifier', base_model)])

    random_search = RandomizedSearchCV(
        estimator=tuning_pipeline, param_distributions=param_dist,
        n_iter=N_TUNING_ITER, cv=cv, scoring='accuracy', n_jobs=-1,
        random_state=42, verbose=1
    )

    try:
        random_search.fit(X, y_encoded)
        print(f"\nBest CV Score for {name}: {random_search.best_score_:.4f}")
        print(f"Best Parameters for {name}:")
        best_params_cleaned = {k.split('__', 1)[1]: v for k, v in random_search.best_params_.items()}
        print(best_params_cleaned)

        # Check if this model is the best overall found so far
        if random_search.best_score_ > best_overall_score:
            best_overall_score = random_search.best_score_
            best_pipeline = random_search.best_estimator_ # Store the best pipeline
            best_tuned_model_name = name
            print(f"==> New overall best model: {name} (CV Score: {best_overall_score:.4f})")

    except Exception as e:
        print(f"\nError during Hyperparameter Tuning for {name}: {e}")
        traceback.print_exc()


print(f"\n--- Finished Tuning ---")
if best_pipeline:
    print(f"Selected best overall model: {best_tuned_model_name} with CV score: {best_overall_score:.4f}")
else:
    print("Error: No model was successfully tuned. Exiting.")
    # Option: Fallback to untuned best model from initial CV if needed
    # fallback_name = max(results, key=results.get)
    # print(f"Falling back to untuned {fallback_name}")
    # best_pipeline = Pipeline([('preprocess', preprocessor), ('classifier', models_to_evaluate[fallback_name])])
    # best_pipeline.fit(X, y_encoded)
    exit()


# --- 8. Final Evaluation on Training Data (Check Overfitting) ---
print(f"\n--- Evaluating Final Model ({best_tuned_model_name}) on Training Data ---")
# This is just a check, the CV score is more important for generalization
try:
    y_pred_train = best_pipeline.predict(X)
    final_train_accuracy = accuracy_score(y_encoded, y_pred_train)
    print(f"Final Model Training Accuracy: {final_train_accuracy:.4f}") # Expect this to be lower than 1.0 now
    if final_train_accuracy == 1.0:
        print("Warning: Final model still achieves 1.0 accuracy on training data. Risk of overfitting remains.")
    print("\nClassification Report (Train - Final Model):\n", classification_report(y_encoded, y_pred_train, target_names=label_encoder.classes_))
except Exception as e:
    print(f"Error during final evaluation on training data: {e}")


# --- 9. Prediction on Test Set ---
print("\n--- Predicting on Test Data ---")
print("Aligning test columns with training columns...")
# (Alignment logic remains the same)
X_test_aligned = X_test.copy()
train_cols_final = X.columns
cols_to_drop = [col for col in X_test_aligned.columns if col not in train_cols_final]
if cols_to_drop: print(f"Dropping columns from test set: {cols_to_drop}"); X_test_aligned = X_test_aligned.drop(columns=cols_to_drop)
cols_to_add = [col for col in train_cols_final if col not in X_test_aligned.columns]
if cols_to_add: print(f"Adding missing columns to test set: {cols_to_add}");
for col in cols_to_add: X_test_aligned[col] = np.nan
X_test_aligned = X_test_aligned[train_cols_final]
print(f"Test data aligned. Final shape for prediction: {X_test_aligned.shape}")

try:
    print(f"Making predictions with the best pipeline ({best_tuned_model_name})...")
    predictions_encoded = best_pipeline.predict(X_test_aligned)
    predictions = label_encoder.inverse_transform(predictions_encoded)
    print("Predictions generated.")

    # --- 10. Create Submission File ---
    print(f"\n--- Creating Submission File ({SUBMISSION_FILE}) ---")
    submission_df = pd.DataFrame({'Index': test_indices, 'Onshore/Offshore': predictions})
    submission_df.columns = ['Index', 'Onshore/Offshore']
    submission_df.to_csv(SUBMISSION_FILE, index=False)
    print(f"Submission file created successfully: {SUBMISSION_FILE}")
    print("\nSubmission file head:"); print(submission_df.head())

except Exception as e:
    print(f"\nError during prediction or submission file creation: {e}"); traceback.print_exc()

print("\n--- Script Finished ---")

