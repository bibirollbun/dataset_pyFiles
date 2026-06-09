import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import warnings
warnings.simplefilter('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


# Feature Generation Function
def create_features(df):
    df = df.copy()

    # Cleanse the 'Sex' column (perform early in preprocessing to prevent NaN during BMR calculation)
    # This ensures the BMR calculation function doesn't need to consider values other than Male/Female
    df['Sex'] = df['Sex'].astype(str).str.strip().str.upper()

    # 1. Body Composition Features
    df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2
    df['BSA'] = 0.007184 * (df['Weight']**0.425) * (df['Height']**0.725) # Body Surface Area (BSA) -> Calculated using Du Bois Formula

    # 2. Basal Metabolic Rate (BMR) - Mifflin-St Jeor Equation -> Apply different formulas based on sex
    def calculate_bmr(row):
        # Assumes 'Sex' is 'MALE' or 'FEMALE' at this point (after cleansing)
        if row['Sex'] == 'MALE': # Match cleansed name
            return 10 * row['Weight'] + 6.25 * row['Height'] - 5 * row['Age'] + 5
        elif row['Sex'] == 'FEMALE': # Match cleansed name
            return 10 * row['Weight'] + 6.25 * row['Height'] - 5 * row['Age'] - 161
        else:
            # If this 'else' block is reached, it means sex cleansing was incomplete
            print(f"DEBUG: Unexpected Sex value '{row['Sex']}' detected during BMR calculation.")
            return np.nan

    df['BMR'] = df.apply(calculate_bmr, axis=1) # Estimated Basal Metabolic Rate (BMR)

    # 3. Exercise-related Features
    df['Max_Heart_Rate'] = 220 - df['Age']
    # Potential for NaN/inf in Heart_Rate_Zone calculation
    df['Heart_Rate_Zone'] = df['Heart_Rate'] / df['Max_Heart_Rate']
    df['Heart_Duration'] = df['Heart_Rate'] * df['Duration']

    # 4. Body State Features
    df['Age_squared'] = df['Age'] ** 2
    df['Temp_deviation'] = df['Body_Temp'] - 36.5
    # Potential for NaN in Age_Group calculation
    df['Age_Group'] = (df['Age'] // 10) * 10
    
    # 5. Interaction Features
    df['Duration_x_Heart_Rate'] = df['Duration'] * df['Heart_Rate']
    # Removed or renamed duplicate definitions
    df['Duration_x_Weight'] = df['Duration'] * df['Weight']
    df['BMI_x_Duration'] = df['BMI'] * df['Duration']
    df['BMR_x_Duration'] = df['BMR'] * df['Duration']

    return df



# Data Preprocessing
def preprocess_data(train_df, test_df):
    # Feature Generation
    train_df = create_features(train_df)
    test_df = create_features(test_df)

    print("\n--- DataFrame Info Immediately After Feature Generation (Train) ---")
    print(train_df.info())
    print("\n--- Missing Values Check Immediately After Feature Generation (Train) ---")
    print(train_df.isnull().sum())
    print("\n--- DataFrame Info Immediately After Feature Generation (Test) ---")
    print(test_df.info())
    print("\n--- Missing Values Check Immediately After Feature Generation (Test) ---")
    print(test_df.isnull().sum())
    print("-" * 50)

    # --- Impute Numerical Features ---
    # Identify numerical columns
    # Ensure BMR_x_Duration is included here
    numerical_cols_train = train_df.select_dtypes(include=np.number).columns.tolist()
    numerical_cols_test = test_df.select_dtypes(include=np.number).columns.tolist()

    # Exclude target variable and ID from imputation
    if 'Calories' in numerical_cols_train:
        numerical_cols_train.remove('Calories')
    if 'id' in numerical_cols_train:
        numerical_cols_train.remove('id')
    if 'id' in numerical_cols_test:
        numerical_cols_test.remove('id')
    
    # Debug: Check column list passed to Imputer
    print("\nDEBUG: Numerical columns passed to Imputer for train_df:")
    print(numerical_cols_train)
    print("\nDEBUG: Numerical columns passed to Imputer for test_df:")
    print(numerical_cols_test)
    print("-" * 50)


    # Initialize Imputer (impute with median)
    imputer = SimpleImputer(strategy='median')

    # Impute numerical columns in train_df
    # Use .loc explicitly to ensure modification of the original DataFrame, not a copy
    # fit_transform returns a NumPy array, so reassign using original column names
    train_df.loc[:, numerical_cols_train] = imputer.fit_transform(train_df[numerical_cols_train])
    # Impute numerical columns in test_df (using imputer learned from train_df)
    test_df.loc[:, numerical_cols_test] = imputer.transform(test_df[numerical_cols_test])

    print("\n--- Missing Values Check After Numerical Feature Imputation (Train) ---")
    print(train_df.isnull().sum())
    print("\n--- Missing Values Check After Numerical Feature Imputation (Test) ---")
    print(test_df.isnull().sum())
    print("-" * 50)

    # One-Hot Encode Categorical Variable (Sex)
    # fit on train_df only, transform on test_df
    # Assumes no NaNs in Sex column after previous cleansing
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    
    # Debug: Check unique values in Sex column
    print(f"\nDEBUG: Train Sex unique values before encoding: {train_df['Sex'].unique()}")
    print(f"DEBUG: Test Sex unique values before encoding: {test_df['Sex'].unique()}")

    # If Sex still has NaNs, imputer should be applied here for categorical data
    # train_df['Sex'].fillna(train_df['Sex'].mode()[0], inplace=True)
    # test_df['Sex'].fillna(train_df['Sex'].mode()[0], inplace=True)


    train_sex_encoded = encoder.fit_transform(train_df[['Sex']])
    test_sex_encoded = encoder.transform(test_df[['Sex']])
    sex_encoded_cols = encoder.get_feature_names_out(['Sex'])

    print(f"DEBUG: Encoded columns: {sex_encoded_cols}")
    print(f"DEBUG: train_sex_encoded shape: {train_sex_encoded.shape}")
    print(f"DEBUG: test_sex_encoded shape: {test_sex_encoded.shape}")


    # Create encoded DataFrames and concatenate with original DataFrame
    train_df_encoded = pd.DataFrame(train_sex_encoded, columns=sex_encoded_cols, index=train_df.index)
    test_df_encoded = pd.DataFrame(test_sex_encoded, columns=sex_encoded_cols, index=test_df.index)

    train_df = pd.concat([train_df.drop(columns=['Sex']), train_df_encoded], axis=1)
    test_df = pd.concat([test_df.drop(columns=['Sex']), test_df_encoded], axis=1)

    return train_df, test_df

# Execute Preprocessing
train_processed, test_processed = preprocess_data(train, test)

# Log transform the target variable
train_processed['Calories_log'] = np.log1p(train_processed['Calories'])

# Feature list
features = [col for col in train_processed.columns if col not in ['id', 'Calories', 'Calories_log']]
X = train_processed[features]
y = train_processed['Calories_log']
X_test = test_processed[features]

print("\n--- First few rows and missing values of final train_processed ---")
print(train_processed.head())
print("\nMissing values in final train_processed:")
print(train_processed.isnull().sum())

print("\n--- First few rows and missing values of final test_processed ---")
print(test_processed.head())
print("\nMissing values in final test_processed:")
print(test_processed.isnull().sum())


import lightgbm as lgb 

# --- Training & Prediction Function (5-fold CV + Log Output) ---
def train_and_predict(X, y, X_test, label=""):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    preds = np.zeros(len(X_test))
    rmse_list = []

    print(f"\n----- Starting {label} Model Training -----")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # LGBMRegressor
        model = lgb.LGBMRegressor(random_state=42)
        
        # Add early stopping
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='rmse', # LightGBM evaluation metric
                  callbacks=[lgb.early_stopping(100, verbose=False)]) # Stop if no improvement for 100 rounds

        val_preds = model.predict(X_val)
        rmse = mean_squared_error(y_val, val_preds, squared=False)
        rmse_list.append(rmse)
        print(f"Fold {fold} RMSE (log): {rmse:.4f}")

        # Sum predictions for test data
        preds += model.predict(X_test) / kf.n_splits # Divide by kf.n_splits to average over folds

    print(f"{label} Model Average RMSE (log): {np.mean(rmse_list):.4f}")
    return preds

# Train a single model on all data
final_preds_log = train_and_predict(X, y, X_test, label="Full Data Single Model (LGBM)")

# Revert log transformation and clip negative values to 0
final_preds = np.expm1(final_preds_log)
final_preds = np.clip(final_preds, 0, None)

print("\n--- Checking Final Predictions ---")
print(final_preds[:5]) # First 5 predicted values


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import lightgbm as lgb
import xgboost as xgb # Import XGBoost
from catboost import CatBoostRegressor, Pool # Import CatBoost

# --- Training & Prediction Function (5-fold CV + Log Output) ---
# Added model_name argument to switch between models
def train_and_predict(X, y, X_test, model_name="LGBM", label=""):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    preds = np.zeros(len(X_test))
    rmse_list = []

    print(f"\n----- Starting {label} Model Training ({model_name}) -----")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = None # Initialize the model

        if model_name == "LGBM":
            model = lgb.LGBMRegressor(random_state=42)
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      eval_metric='rmse',
                      callbacks=[lgb.early_stopping(100, verbose=False)])

        elif model_name == "XGBoost":
            # Example XGBoost parameters
            model = xgb.XGBRegressor(
                objective='reg:squarederror', # Objective function for regression
                n_estimators=1000,            # Number of trees (set high and adjust with early stopping)
                learning_rate=0.05,           # Learning rate
                random_state=42,
                n_jobs=-1,                    # Use all available CPU cores
                tree_method='hist',           # Use histogram-based algorithm for speed
                eval_metric='rmse'            # Evaluation metric
            )
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      early_stopping_rounds=100, # Stop if no improvement for 100 rounds
                      verbose=False)

        elif model_name == "CatBoost":
            # Example CatBoost parameters
            model = CatBoostRegressor(
                iterations=1000,              # Number of trees (equivalent to n_estimators in LGBM/XGBoost)
                learning_rate=0.05,           # Learning rate
                random_seed=42,               # Seed value
                eval_metric='RMSE',           # Evaluation metric
                verbose=False,                # Suppress verbose output during training
                early_stopping_rounds=100,    # Stop if no improvement for 100 rounds
                # CatBoost can handle categorical features directly, so the 'Sex' column
                # could be kept un-One-Hot Encoded in preprocess_data and included in features.
                # However, since 'Sex' is currently One-Hot Encoded, it fits into numerical_cols_train.
                # categorical_features_indices=[] # Specify indices of categorical features
            )
            # CatBoost commonly uses Pool objects for evaluation sets
            train_pool = Pool(X_train, y_train)
            val_pool = Pool(X_val, y_val)
            model.fit(train_pool,
                      eval_set=val_pool,
                      early_stopping_rounds=100,
                      verbose=False)
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        val_preds = model.predict(X_val)
        rmse = mean_squared_error(y_val, val_preds, squared=False)
        rmse_list.append(rmse)
        print(f"Fold {fold} RMSE (log): {rmse:.4f}")

        # Sum predictions for test data
        preds += model.predict(X_test) / kf.n_splits

    print(f"{label} Model Average RMSE (log): {np.mean(rmse_list):.4f}")
    return preds

# --- Model Comparison ---
# LGBM
final_preds_log_lgbm = train_and_predict(X, y, X_test, model_name="LGBM", label="LGBM")
final_preds_lgbm = np.expm1(final_preds_log_lgbm)
final_preds_lgbm = np.clip(final_preds_lgbm, 0, None)
print("\n--- LGBM Final Predictions Check ---")
print(final_preds_lgbm[:5])

# XGBoost
final_preds_log_xgb = train_and_predict(X, y, X_test, model_name="XGBoost", label="XGBoost")
final_preds_xgb = np.expm1(final_preds_log_xgb)
final_preds_xgb = np.clip(final_preds_xgb, 0, None)
print("\n--- XGBoost Final Predictions Check ---")
print(final_preds_xgb[:5])

# CatBoost
final_preds_log_cat = train_and_predict(X, y, X_test, model_name="CatBoost", label="CatBoost")
final_preds_cat = np.expm1(final_preds_log_cat)
final_preds_cat = np.clip(final_preds_cat, 0, None)
print("\n--- CatBoost Final Predictions Check ---")
print(final_preds_cat[:5])

# Compare results or ensemble (e.g., average the predictions of the 3 models)
final_ensemble_preds = (final_preds_lgbm + final_preds_xgb + final_preds_cat) / 3
print("\n--- Ensemble Predictions Check ---")
print(final_ensemble_preds[:5])


# Create submission file
submission = sample_submission.copy()
submission['Calories'] = final_preds
submission.to_csv("submission.csv", index=False)

print("submission.csv has been created.")




