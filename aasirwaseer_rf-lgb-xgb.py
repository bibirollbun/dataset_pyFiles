# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings
warnings.filterwarnings("ignore")


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv("/kaggle/input/terrain-prices-reggression/train.csv")
test = pd.read_csv("/kaggle/input/terrain-prices-reggression/test.csv")
sample_submission = pd.read_csv("/kaggle/input/terrain-prices-reggression/sample_submission.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()



# Check for missing values
print("Missing values in train:\n", train.isnull().sum().sort_values(ascending=False).head())
print("\nMissing values in test:\n", test.isnull().sum().sort_values(ascending=False).head())

# Data types
print("\nData types:\n", train.dtypes.value_counts())

# Target variable summary
print("\nTarget variable stats:")
print(train['target'].describe())



from sklearn.preprocessing import LabelEncoder

# Drop ID column
train = train.drop("id", axis=1)
test = test.drop("id", axis=1)

# Separate target variable
y = train["target"]
X = train.drop("target", axis=1)

# Identify categorical columns
cat_cols = X.select_dtypes(include="object").columns.tolist()
print("Categorical columns:", cat_cols)

# Label encode categorical features (for simplicity)
le = LabelEncoder()
for col in cat_cols:
    X[col] = le.fit_transform(X[col])
    test[col] = le.transform(test[col])



from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import numpy as np

# Initialize model
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# Evaluate using cross-validation
scores = cross_val_score(rf, X, y, scoring="neg_root_mean_squared_error", cv=5)

print("CV RMSE scores:", -scores)
print("Mean CV RMSE:", -scores.mean())



# Train on full training set
rf.fit(X, y)

# Predict on test set
test_preds = rf.predict(test)

# Prepare submission
sample_submission["target"] = test_preds
sample_submission.to_csv("submission.csv", index=False)

print("Submission file saved as submission.csv")



from sklearn.model_selection import RandomizedSearchCV

# Define the parameter grid
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Setup RandomizedSearchCV
rf_random = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=25,
    cv=3,
    verbose=2,
    random_state=42,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)

# Run tuning
rf_random.fit(X, y)

# Best parameters and score
print("Best Params:", rf_random.best_params_)
print("Best CV Score (neg RMSE):", rf_random.best_score_)



# Train model with best parameters
best_rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=30,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='auto',
    random_state=42,
    n_jobs=-1
)

# Fit on full training data
best_rf.fit(X, y)

# Predict on test set
tuned_preds = best_rf.predict(test)

# Create new submission
sample_submission["target"] = tuned_preds
sample_submission.to_csv("submission_tuned.csv", index=False)

print("Tuned submission saved as submission_tuned.csv")



import lightgbm as lgb
from sklearn.model_selection import cross_val_score

# Initialize LightGBM Regressor
lgb_model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=-1,
    random_state=42,
    n_jobs=-1
)

# Cross-validate
lgb_scores = cross_val_score(lgb_model, X, y, cv=5, scoring="neg_root_mean_squared_error")

print("CV RMSE scores:", -lgb_scores)
print("Mean CV RMSE:", -lgb_scores.mean())



# Train on full data
lgb_model.fit(X, y)

# Predict on test set
lgb_preds = lgb_model.predict(test)

# Save submission
sample_submission["target"] = lgb_preds
sample_submission.to_csv("submission_lgb.csv", index=False)

print("LightGBM submission saved as submission_lgb.csv")



from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb

# Parameter grid for tuning
param_dist = {
    'n_estimators': [300, 500, 700, 1000],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'num_leaves': [20, 31, 40, 60],
    'max_depth': [-1, 5, 10, 20],
    'min_child_samples': [5, 10, 20],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0, 0.1, 0.5]
}

# Setup RandomizedSearchCV
lgb_random = RandomizedSearchCV(
    estimator=lgb.LGBMRegressor(random_state=42, n_jobs=-1),
    param_distributions=param_dist,
    n_iter=30,  # You can increase if time allows
    scoring='neg_root_mean_squared_error',
    cv=3,
    verbose=2,
    random_state=42
)

# Run the tuning
lgb_random.fit(X, y)

# Output best parameters
print("Best Parameters:", lgb_random.best_params_)
print("Best CV Score (neg RMSE):", lgb_random.best_score_)



# Train tuned LightGBM model
best_lgb = lgb.LGBMRegressor(
    subsample=1.0,
    reg_lambda=0,
    reg_alpha=0.5,
    num_leaves=20,
    n_estimators=700,
    min_child_samples=10,
    max_depth=10,
    learning_rate=0.01,
    colsample_bytree=1.0,
    random_state=42,
    n_jobs=-1
)

# Fit on full training data
best_lgb.fit(X, y)

# Predict on test set
final_preds = best_lgb.predict(test)

# Save submission
sample_submission["target"] = final_preds
sample_submission.to_csv("submission_lgb_tuned.csv", index=False)

print("Tuned LightGBM submission saved as submission_lgb_tuned.csv")



from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

# Initialize baseline XGBoost model
xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    n_jobs=-1
)

# Evaluate with CV
xgb_scores = cross_val_score(xgb, X, y, cv=5, scoring="neg_root_mean_squared_error")

print("CV RMSE scores:", -xgb_scores)
print("Mean CV RMSE:", -xgb_scores.mean())



from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

# Define parameter grid
xgb_param_grid = {
    'n_estimators': [300, 500, 700],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'max_depth': [3, 5, 7, 10],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0, 0.1, 0.5],
    'min_child_weight': [1, 3, 5]
}

xgb_random = RandomizedSearchCV(
    estimator=XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
    param_distributions=xgb_param_grid,
    n_iter=30,
    scoring='neg_root_mean_squared_error',
    cv=3,
    verbose=2,
    random_state=42
)

xgb_random.fit(X, y)

print("Best Parameters:", xgb_random.best_params_)
print("Best CV Score (neg RMSE):", xgb_random.best_score_)



import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# Load data
train = pd.read_csv('/kaggle/input/terrain-prices-regression/train.csv')
test = pd.read_csv('/kaggle/input/terrain-prices-regression/test.csv')
submission = pd.read_csv('/kaggle/input/terrain-prices-regression/sample_submission.csv')

# Backup IDs
test_ids = test['id']

# Drop ID columns
train.drop(columns=['id'], inplace=True)
test.drop(columns=['id'], inplace=True)

# Concatenate train+test for joint processing
target = train['price_per_m2']
train.drop(columns=['price_per_m2'], inplace=True)
full = pd.concat([train, test], axis=0).reset_index(drop=True)

# Basic feature cleanup
cat_cols = full.select_dtypes(include='object').columns.tolist()
num_cols = full.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Imputation
full[cat_cols] = full[cat_cols].fillna('Missing')
full[num_cols] = full[num_cols].fillna(full[num_cols].median())

# Interaction Features (only if all involved columns exist)
interaction_pairs = [
    ('elevation', 'slope_deg', 'elev_slope'),
    ('soil_quality', 'amenities_score', 'soil_amen'),
    ('crime_rate', 'median_income_area', 'crime_income'),
    ('land_area_m2', 'soil_quality', 'land_soil')
]

for col1, col2, new_col in interaction_pairs:
    if col1 in full.columns and col2 in full.columns:
        full[new_col] = full[col1] * full[col2]

# Categorical Encoding
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
full[cat_cols] = encoder.fit_transform(full[cat_cols])

# Polynomial Features
poly_feats = ['slope_deg', 'soil_quality', 'amenities_score', 'median_income_area', 'crime_rate']
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_data = poly.fit_transform(full[poly_feats])
poly_feature_names = poly.get_feature_names_out(poly_feats)
poly_df = pd.DataFrame(poly_data, columns=poly_feature_names)
poly_df.index = full.index

# Drop original poly columns to avoid duplication
full.drop(columns=poly_feats, inplace=True)
full = pd.concat([full, poly_df], axis=1)

# Split back to train and test
X_train = full.iloc[:len(target), :]
X_test = full.iloc[len(target):, :]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# Cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(xgb, X_train_scaled, target, cv=cv, scoring='r2')
print(f"Mean CV R2 Score: {np.mean(cv_scores):.5f}")

# Train on full data
xgb.fit(X_train_scaled, target)

# Predict on test
preds = xgb.predict(X_test_scaled)

# Submission
submission['price_per_m2'] = preds
submission.to_csv('submission_xgb_poly_tuned.csv', index=False)



import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, KFold

train_df = pd.read_csv('/kaggle/input/terrain-prices-reggression/train.csv')
test_df = pd.read_csv('/kaggle/input/terrain-prices-reggression/test.csv')

# Separate target
X = train_df.drop(columns=["target"])
y = train_df["target"]
X_test = test_df.copy()

# Identify features
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Build numerical pipeline with polynomial features
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler())
])

# Build categorical pipeline
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine using ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])

# Apply preprocessing
X_processed = preprocessor.fit_transform(X)
X_test_processed = preprocessor.transform(X_test)  # ❗Fixed: no 'target' column assumed in test

# Define model
xgb_final = XGBRegressor(
    subsample=1.0,
    reg_lambda=0,
    reg_alpha=0,
    n_estimators=500,
    min_child_weight=5,
    max_depth=3,
    learning_rate=0.03,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# Cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(xgb_final, X_processed, y, scoring='neg_root_mean_squared_error', cv=cv)
print("Mean CV RMSE:", -np.mean(scores))

# Fit on full training data
xgb_final.fit(X_processed, y)

# Predict on test set
preds = xgb_final.predict(X_test_processed)

# Prepare submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'target': preds
})
submission.to_csv('submission_xgb_poly_tuned.csv', index=False)





