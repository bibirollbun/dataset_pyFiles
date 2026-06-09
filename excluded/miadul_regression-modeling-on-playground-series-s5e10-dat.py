# Step 2: Import Libraries

# Basic libraries
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Regression Models
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Warnings
import warnings
warnings.filterwarnings("ignore")

print("Libraries Imported Successfully!")



# Step 3: Load Dataset

# File paths
train_path = "/kaggle/input/playground-series-s5e10/train.csv"
test_path = "/kaggle/input/playground-series-s5e10/test.csv"
sample_path = "/kaggle/input/playground-series-s5e10/sample_submission.csv"

# Load datasets
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_df = pd.read_csv(sample_path)

# Display basic info
print("Train Shape:", train_df.shape)
print("Test Shape:", test_df.shape)
print("Sample Submission Shape:", sample_df.shape)

# Preview top rows
print("\nTrain Preview:")
display(train_df.head())

print("\nTest Preview:")
display(test_df.head())

print("\nSample Submission Preview:")
display(sample_df.head())



# Step 4: Exploratory Data Analysis (EDA)

print("===== Basic Information =====")
display(train_df.info())

print("\n===== Summary Statistics =====")
display(train_df.describe())

# -----------------------------
# 1. Check Missing Values
# -----------------------------
print("\n===== Missing Values =====")
missing_values = train_df.isnull().sum().sort_values(ascending=False)
display(missing_values)

# Visual missing value heatmap
plt.figure(figsize=(10, 5))
sns.heatmap(train_df.isnull(), cbar=False)
plt.title("Missing Values Heatmap")
plt.show()





# 2. Target Variable Distribution
# -----------------------------
target = train_df.columns[-1]   # last column as target
plt.figure(figsize=(7, 4))
sns.histplot(train_df[target], kde=True)
plt.title(f"Distribution of Target: {target}")
plt.xlabel(target)
plt.ylabel("Count")
plt.show()


# Step 5: Data Cleaning

print("===== Before Cleaning =====")
print("Missing values:\n", train_df.isnull().sum())

# ------------------------------------------------------
# 1. Handle Missing Values
# ------------------------------------------------------

# Numerical columns → fill with median
num_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    train_df[col].fillna(train_df[col].median(), inplace=True)
    if col in test_df.columns:
        test_df[col].fillna(train_df[col].median(), inplace=True)

# Categorical columns → fill with mode
cat_cols = train_df.select_dtypes(include=['object', 'category']).columns
for col in cat_cols:
    train_df[col].fillna(train_df[col].mode()[0], inplace=True)
    if col in test_df.columns:
        test_df[col].fillna(train_df[col].mode()[0], inplace=True)

# ------------------------------------------------------
# 2. Remove Duplicates
# ------------------------------------------------------
before = train_df.shape[0]
train_df.drop_duplicates(inplace=True)
after = train_df.shape[0]
print(f"Duplicates removed: {before - after}")

# ------------------------------------------------------
# 3. Fix Data Types (if needed)
# ------------------------------------------------------
# Example : converting object to category
for col in cat_cols:
    train_df[col] = train_df[col].astype('category')
    if col in test_df.columns:
        test_df[col] = test_df[col].astype('category')

# ------------------------------------------------------
# 4. Outlier Handling (Optional)
# ------------------------------------------------------
# Using IQR Clipping for numeric columns
for col in num_cols:
    Q1 = train_df[col].quantile(0.25)
    Q3 = train_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    train_df[col] = np.clip(train_df[col], lower_bound, upper_bound)

print("===== After Cleaning =====")
print(train_df.info())



# Step 6: Feature Engineering

print("===== Feature Engineering Started =====")

# ------------------------------------------------------
# 1. Separate Target Column
# ------------------------------------------------------
target = train_df.columns[-1]   # assuming last column is target
X = train_df.drop(columns=[target])
y = train_df[target]

# ------------------------------------------------------
# 2. Identify Numeric & Categorical Columns
# ------------------------------------------------------
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns

print("Numeric Columns:", list(numeric_cols))
print("Categorical Columns:", list(categorical_cols))

# ------------------------------------------------------
# 3. Label Encoding for Categorical Features
# ------------------------------------------------------
from sklearn.preprocessing import LabelEncoder

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    test_df[col] = le.transform(test_df[col])
    label_encoders[col] = le

print("Label Encoding Done.")

# ------------------------------------------------------
# 4. Feature Scaling (StandardScaler)
# -----



# Step 7: Train-Test Split

from sklearn.model_selection import train_test_split

# Split data into training and validation sets
# 80% train, 20% validation
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Check shapes
print("X_train shape:", X_train.shape)
print("X_valid shape:", X_valid.shape)
print("y_train shape:", y_train.shape)
print("y_valid shape:", y_valid.shape)



# Step 8: Baseline Regression Models

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Function to evaluate models
def evaluate_model(model, X_train, y_train, X_valid, y_valid):
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_valid = model.predict(X_valid)
    
    print(f"Model: {model.__class__.__name__}")
    print("Train MAE:", round(mean_absolute_error(y_train, y_pred_train), 4))
    print("Train RMSE:", round(mean_squared_error(y_train, y_pred_train, squared=False), 4))
    print("Train R2:", round(r2_score(y_train, y_pred_train), 4))
    print("Validation MAE:", round(mean_absolute_error(y_valid, y_pred_valid), 4))
    print("Validation RMSE:", round(mean_squared_error(y_valid, y_pred_valid, squared=False), 4))
    print("Validation R2:", round(r2_score(y_valid, y_pred_valid), 4))
    print("-" * 50)

# -----------------------------
# 1. Linear Regression
# -----------------------------
lr = LinearRegression()
evaluate_model(lr, X_train, y_train, X_valid, y_valid)

# -----------------------------
# 2. Ridge Regression
# -----------------------------
ridge = Ridge(alpha=1.0)
evaluate_model(ridge, X_train, y_train, X_valid, y_valid)

# -----------------------------
# 3. Lasso Regression
# -----------------------------
lasso = Lasso(alpha=0.01)
evaluate_model(lasso, X_train, y_train, X_valid, y_valid)

# -----------------------------
# 4. ElasticNet Regression
# -----------------------------
elastic = ElasticNet(alpha=0.01, l1_ratio=0.5)
evaluate_model(elastic, X_train, y_train, X_valid, y_valid)



# Step 9: Tree-Based Regression Models

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Function to evaluate models (reuse from Step 8)
def evaluate_model(model, X_train, y_train, X_valid, y_valid):
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_valid = model.predict(X_valid)
    
    print(f"Model: {model.__class__.__name__}")
    print("Train MAE:", round(mean_absolute_error(y_train, y_pred_train), 4))
    print("Train RMSE:", round(mean_squared_error(y_train, y_pred_train, squared=False), 4))
    print("Train R2:", round(r2_score(y_train, y_pred_train), 4))
    print("Validation MAE:", round(mean_absolute_error(y_valid, y_pred_valid), 4))
    print("Validation RMSE:", round(mean_squared_error(y_valid, y_pred_valid, squared=False), 4))
    print("Validation R2:", round(r2_score(y_valid, y_pred_valid), 4))
    print("-" * 50)

# -----------------------------
# 1. Random Forest Regressor
# -----------------------------
rf = RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42)
evaluate_model(rf, X_train, y_train, X_valid, y_valid)

# -----------------------------
# 2. Gradient Boosting Regressor
# -----------------------------
gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
evaluate_model(gbr, X_train, y_train, X_valid, y_valid)

# -----------------------------
# 3. XGBoost Regressor
# -----------------------------
xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42, verbosity=0)
evaluate_model(xgb, X_train, y_train, X_valid, y_valid)

# -----------------------------
# 4. LightGBM Regressor
# -----------------------------
lgbm = LGBMRegressor(n_estimators=200, learning_rate=0.1, max_depth=-1, random_state=42)
evaluate_model(lgbm, X_train, y_train, X_valid, y_valid)

# -----------------------------
# 5. CatBoost Regressor
# -----------------------------
cat = CatBoostRegressor(n_estimators=200, learning_rate=0.1, depth=6, random_state=42, verbose=0)
evaluate_model(cat, X_train, y_train, X_valid, y_valid)



# Step 10: Hyperparameter Tuning

from sklearn.model_selection import RandomizedSearchCV

# Example: Hyperparameter tuning for RandomForestRegressor
rf = RandomForestRegressor(random_state=42)

# Hyperparameter grid
param_dist = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# RandomizedSearchCV
rf_random = RandomizedSearchCV(
    estimator=rf, 
    param_distributions=param_dist, 
    n_iter=20,        # number of random combinations
    cv=3,             # 3-fold cross-validation
    scoring='neg_root_mean_squared_error', 
    verbose=2, 
    random_state=42,
    n_jobs=-1
)

# Fit on training data
rf_random.fit(X_train, y_train)

# Best parameters
print("Best Parameters:", rf_random.best_params_)

# Evaluate on validation set
best_rf = rf_random.best_estimator_
y_pred_valid = best_rf.predict(X_valid)

print("\nValidation Performance of Tuned RF:")
print("MAE:", round(mean_absolute_error(y_valid, y_pred_valid), 4))
print("RMSE:", round(mean_squared_error(y_valid, y_pred_valid, squared=False), 4))
print("R2:", round(r2_score(y_valid, y_pred_valid), 4))



# Step 11: Train Final Model on Full Training Set

# Using the best tuned RandomForestRegressor from Step 10
final_model = rf_random.best_estimator_

# Fit on full training data (train + validation)
X_full = pd.concat([X_train, X_valid], axis=0)
y_full = pd.concat([y_train, y_valid], axis=0)

final_model.fit(X_full, y_full)

# Check training performance
y_pred_full = final_model.predict(X_full)
print("===== Final Model Performance on Full Training Data =====")
print("MAE:", round(mean_absolute_error(y_full, y_pred_full), 4))
print("RMSE:", round(mean_squared_error(y_full, y_pred_full, squared=False), 4))
print("R2:", round(r2_score(y_full, y_pred_full), 4))



# Step 12: Predict on Test Set

# Ensure test_df has the same features as training data
X_test = test_df[X_full.columns]

# Predict using the final model
test_predictions = final_model.predict(X_test)

# Preview predictions
print("Test Predictions Preview:")
print(test_predictions[:10])



# Step 13: Create Submission File

# Copy the sample submission file
submission = sample_df.copy()

# Assuming the target column name in sample_submission is same as in train_df
submission[target] = test_predictions

# Save to CSV
submission_file = "submission.csv"
submission.to_csv(submission_file, index=False)

print(f"Submission file '{submission_file}' created successfully!")
display(submission.head())


