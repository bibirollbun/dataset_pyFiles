print("Done")


import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import QuantileTransformer

# Import callbacks for early stopping
import lightgbm as lgb
import xgboost
from xgboost.callback import EarlyStopping as XgbEarlyStopping


# Load Data
TRAIN_DATA_PATH = "/kaggle/input/playground-series-s5e5/train.csv"
TEST_DATA_PATH = "/kaggle/input/playground-series-s5e5/test.csv"
SAMPLE_SUBMISSION_PATH = "/kaggle/input/playground-series-s5e5/sample_submission.csv"

train_df = pd.read_csv(TRAIN_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)
sample_submission_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)

# Combine
train_ids = train_df['id']
test_ids = test_df['id']
train_calories = train_df['Calories']

train_df = train_df.drop(['id', 'Calories'], axis=1)
test_df = test_df.drop('id', axis=1)

combined_df = pd.concat([train_df, test_df], ignore_index=True)

# Cleaning
cols_with_missing = combined_df.isnull().sum()[combined_df.isnull().sum() > 0].index.tolist()
if cols_with_missing:
    for col in combined_df.isnull().sum()[combined_df.isnull().sum() > 0].index.tolist():
        if combined_df[col].dtype in np.number: combined_df[col].fillna(combined_df[col].median(), inplace=True)
        else: combined_df[col].fillna(combined_df[col].mode()[0], inplace=True)

# --- Enhanced Outlier Handling (Percentile Clipping) ---
numerical_cols_for_outliers = combined_df.select_dtypes(include=np.number).columns.tolist()
for col in numerical_cols_for_outliers:
    if col in combined_df.columns:
        # Use 1st and 99th percentiles for clipping
        lower_bound = combined_df[col].quantile(0.01)
        upper_bound = combined_df[col].quantile(0.99)
        combined_df[col] = np.clip(combined_df[col], lower_bound, upper_bound)


# --- Advanced Feature Engineering ---
combined_df['BMI'] = combined_df['Weight'] / ((combined_df['Height'] / 100) ** 2)
combined_df['Heart_Rate_per_Duration'] = combined_df['Heart_Rate'] / (combined_df['Duration'] + 1e-6)
combined_df['Temp_per_Duration'] = combined_df['Body_Temp'] / (combined_df['Duration'] + 1e-6)
combined_df['Heart_Rate_x_Temp'] = combined_df['Heart_Rate'] * combined_df['Body_Temp']
combined_df['Age_x_Duration'] = combined_df['Age'] * combined_df['Duration']

# Add more interaction terms
combined_df['BMI_x_Duration'] = combined_df['BMI'] * combined_df['Duration']
combined_df['Age_x_Heart_Rate'] = combined_df['Age'] * combined_df['Heart_Rate']
combined_df['Height_x_Weight'] = combined_df['Height'] * combined_df['Weight']

# Add polynomial features (degree 2 for some key features)
for col in ['Age', 'Duration', 'Heart_Rate', 'Body_Temp', 'BMI']:
    combined_df[f'{col}_sq'] = combined_df[col]**2


# Separate data AFTER feature engineering
X = combined_df.iloc[:len(train_df)].reset_index(drop=True)
X_test = combined_df.iloc[len(train_df):].reset_index(drop=True)
y = train_calories.reset_index(drop=True)

# Identify features AFTER feature engineering
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()


# Preprocessing steps (Add PolynomialFeatures for interactions here)
numerical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                       ('scaler', StandardScaler()),
                                       ('transformer', QuantileTransformer(output_distribution='normal')),
                                       ('poly_interactions', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)) # Add interaction features
                                      ])

categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                       ('onehot', OneHotEncoder(handle_unknown='ignore'))
                                      ])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'
)

# Base Models
y_transformed = np.log1p(y)

# Model Params (Reduced n_estimators for speed, with potential GPU acceleration)
# Reduced n_estimators from 6000 to 3500 as a starting point
xgb_params = {'objective': 'reg:squarederror', 'n_estimators': 3500, 'learning_rate': 0.007, 'max_depth': 9, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
              'tree_method': 'hist', 'device': 'cuda'} # Added GPU params
lgb_params = {'objective': 'regression_l1', 'n_estimators': 3500, 'learning_rate': 0.007, 'num_leaves': 300, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
              'device': 'gpu'} # Added GPU param
# Removed n_jobs from cat_params and add task_type='GPU'
cat_params = {'iterations': 3500, 'learning_rate': 0.007, 'depth': 9, 'l2_leaf_reg': 3, 'random_state': 42, 'verbose': 0,
              'task_type': 'GPU'} # Added GPU param, removed n_jobs
ridge_params = {'alpha': 0.8, 'random_state': 42}

# CV Setup
NFOLDS = 5 # Keep 5 folds for now, can reduce to 3 if still too slow
folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)

# OOF/Test arrays
oof_preds_xgb = np.zeros(X.shape[0])
oof_preds_lgb = np.zeros(X.shape[0])
oof_preds_cat = np.zeros(X.shape[0])
oof_preds_ridge = np.zeros(X.shape[0])

test_preds_xgb = np.zeros(X_test.shape[0])
test_preds_lgb = np.zeros(X_test.shape[0])
test_preds_cat = np.zeros(X_test.shape[0])
test_preds_ridge = np.zeros(X_test.shape[0])

print(f"Starting {NFOLDS}-Fold CV...")

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
    print(f"Fold: {fold_}")

    X_train_fold, y_train_fold = X.iloc[trn_idx], y_transformed.iloc[trn_idx]
    X_valid_fold, y_valid_fold = X.iloc[val_idx], y_transformed.iloc[val_idx]
    X_test_fold = X_test

    # Fit Preprocessor on Training Fold and Transform Data
    preprocessor.fit(X_train_fold)
    X_train_transformed = preprocessor.transform(X_train_fold).astype(np.float32)
    X_valid_transformed = preprocessor.transform(X_valid_fold).astype(np.float32)
    X_test_transformed = preprocessor.transform(X_test_fold).astype(np.float32)

    # Train Base Models (on transformed data)
    # XGBoost
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train_transformed, y_train_fold,
                  eval_set=[(X_valid_transformed, y_valid_fold)],
                  callbacks=[XgbEarlyStopping(rounds=150)])
    oof_preds_xgb[val_idx] = xgb_model.predict(X_valid_transformed)
    test_preds_xgb += xgb_model.predict(X_test_transformed) / NFOLDS

    # LightGBM
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X_train_transformed, y_train_fold,
                  eval_set=[(X_valid_transformed, y_valid_fold)],
                  callbacks=[lgb.early_stopping(stopping_rounds=150)])
    oof_preds_lgb[val_idx] = lgb_model.predict(X_valid_transformed)
    test_preds_lgb += lgb_model.predict(X_test_transformed) / NFOLDS

    # CatBoost
    cat_model = CatBoostRegressor(**cat_params)
    cat_model.fit(X_train_transformed, y_train_fold,
                  eval_set=[(X_valid_transformed, y_valid_fold)],
                  early_stopping_rounds=150, verbose=False)
    oof_preds_cat[val_idx] = cat_model.predict(X_valid_transformed)
    test_preds_cat += cat_model.predict(X_test_transformed) / NFOLDS

    # Ridge Base
    ridge_model = Ridge(**ridge_params)
    ridge_model.fit(X_train_transformed, y_train_fold)
    oof_preds_ridge[val_idx] = ridge_model.predict(X_valid_transformed)
    test_preds_ridge += ridge_model.predict(X_test_transformed) / NFOLDS

print("CV complete.")

# --- Stacking Meta (with enhanced meta-features) ---
# Combine base model predictions
X_meta_train_base = np.column_stack((oof_preds_xgb, oof_preds_lgb, oof_preds_cat, oof_preds_ridge))
X_meta_test_base = np.column_stack((test_preds_xgb, test_preds_lgb, test_preds_cat, test_preds_ridge))

# Create meta-features from base model predictions
X_meta_train_engineered = pd.DataFrame({
    'meta_mean': X_meta_train_base.mean(axis=1),
    'meta_median': np.median(X_meta_train_base, axis=1),
    'meta_min': X_meta_train_base.min(axis=1),
    'meta_max': X_meta_train_base.max(axis=1),
    'meta_range': X_meta_train_base.max(axis=1) - X_meta_train_base.min(axis=1),
    'meta_std': X_meta_train_base.std(axis=1)
})

X_meta_test_engineered = pd.DataFrame({
    'meta_mean': X_meta_test_base.mean(axis=1),
    'meta_median': np.median(X_meta_test_base, axis=1),
    'meta_min': X_meta_test_base.min(axis=1),
    'meta_max': X_meta_test_base.max(axis=1),
    'meta_range': X_meta_test_base.max(axis=1) - X_meta_test_base.min(axis=1),
    'meta_std': X_meta_test_base.std(axis=1)
})

# Add pairwise interaction features between base model predictions
# Using PolynomialFeatures for systematic interactions
poly_meta = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

X_meta_train_interactions = poly_meta.fit_transform(X_meta_train_base)
X_meta_test_interactions = poly_meta.transform(X_meta_test_base) # Use transform on test set

# Combine base predictions, engineered features, and interaction features for the meta-model input
X_meta_train = np.hstack((X_meta_train_base, X_meta_train_engineered.values, X_meta_train_interactions))
X_meta_test = np.hstack((X_meta_test_base, X_meta_test_engineered.values, X_meta_test_interactions))


# Meta Model
meta_model = Ridge(alpha=0.5) # Still using Ridge

# Train Meta
print("Train Meta...")
meta_model.fit(X_meta_train, y_transformed)
print("Meta complete.")

# Final Prediction
print("Final Prediction...")
final_predictions_transformed = meta_model.predict(X_meta_test)

# Inverse transform
predictions = np.expm1(final_predictions_transformed)
predictions[predictions < 0] = 0

# Submission
submission_df = pd.DataFrame({'id': test_ids, 'Calories': predictions})
submission_df.to_csv('submission.csv', index=False)

print("Done")

