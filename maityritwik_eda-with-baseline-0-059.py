# Step 1: Load and Inspect Data

# Import required library
import pandas as pd

# Define file paths (already provided earlier)
train_path = "/kaggle/input/playground-series-s5e10/train.csv"
test_path = "/kaggle/input/playground-series-s5e10/test.csv"
# submission_path = "/mnt/data/sample_submission.csv"

# Load the datasets
train_df = pd.read_csv(train_path)   # Training dataset with target BeatsPerMinute
test_df = pd.read_csv(test_path)     # Test dataset without target
# submission_df = pd.read_csv(submission_path)  # Sample submission format

# Inspect shapes of datasets
print("Train shape:", train_df.shape)        # Should include target
print("Test shape:", test_df.shape)          # Should exclude target
# print("Submission shape:", submission_df.shape)  # Should have 2 columns: ID, BeatsPerMinute

# Preview first few rows
print("\nTrain preview:")
print(train_df.head())

print("\nTest preview:")
print(test_df.head())

# print("\nSubmission preview:")
# print(submission_df.head())

# Check for missing values in train and test
print("\nMissing values in Train:")
print(train_df.isnull().sum())

print("\nMissing values in Test:")
print(test_df.isnull().sum())


#  Display column information
# This helps us understand the data types and whether there are missing values.
print(train_df.info())
print(test_df.info())

# Basic statistical summary
# This provides mean, min, max, and percentiles for numerical columns.
print(test_df.describe())
print(test_df.describe())


# Step 2.1: Import visualization libraries
import matplotlib.pyplot as plt   # for plotting
import seaborn as sns             # for advanced plots

# Set style for seaborn plots
sns.set(style="whitegrid")



# Plot distribution of accident_risk
plt.figure(figsize=(8,5))
sns.histplot(train_df['accident_risk'], bins=30, kde=True, color='blue')  # kde adds smooth curve
plt.title("Distribution of Target: accident_risk")
plt.xlabel("Accident Risk")
plt.ylabel("Count")
plt.show()



# Select numeric columns (excluding id)
numeric_features = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']

# Plot histograms for each numeric feature
train_df[numeric_features].hist(figsize=(10,8), bins=20, color='blue', edgecolor='black')
plt.suptitle("Distributions of Numeric Features")
plt.show()



# Boxplots to visualize distribution of accident risk across categories
categorical_features = ['road_type', 'lighting', 'weather', 'time_of_day']

plt.figure(figsize=(14,10))
for i, col in enumerate(categorical_features, 1):
    plt.subplot(2,2,i)  # 2x2 grid of plots
    sns.boxplot(x=col, y='accident_risk', data=train_df)
    plt.xticks(rotation=30)  # rotate labels for readability
    plt.title(f"Accident Risk by {col}")
plt.tight_layout()
plt.show()



# Compute correlation matrix
corr_matrix = train_df[numeric_features + ['accident_risk']].corr()

# Plot heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()



# Step 3.1: Separate features and target
X = train_df.drop(columns=['accident_risk', 'id'])  # Features (remove target + id)
y = train_df['accident_risk']                      # Target variable
X_test = test_df.drop(columns=['id'])              # Test features (no target available)



# Identify column groups
numeric_features = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
categorical_features = ['road_type', 'lighting', 'weather', 'time_of_day']
boolean_features = ['road_signs_present', 'public_road', 'holiday', 'school_season']

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)
print("Boolean features:", boolean_features)



from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Numeric transformer: scale features
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Categorical transformer: one-hot encode features
categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Boolean transformer: convert to int (0/1)
# (Note: Boolean values are already True/False, so we just cast them)
boolean_transformer = 'passthrough'

# Combine all into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('bool', boolean_transformer, boolean_features)
    ])



# Fit and transform the training data
X_preprocessed = preprocessor.fit_transform(X)

# Transform the test data
X_test_preprocessed = preprocessor.transform(X_test)

# Check transformed feature shape
print("Original Train Shape:", X.shape)
print("Transformed Train Shape:", X_preprocessed.shape)
print("Original Test Shape:", X_test.shape)
print("Transformed Test Shape:", X_test_preprocessed.shape)



# Step 4.1: Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np



# Perform an 80-20 train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_preprocessed, y, test_size=0.2, random_state=42)

print("Training set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)



# Initialize and train Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Predict on validation set
y_val_pred_lr = lin_reg.predict(X_val)

# Calculate RMSE
rmse_lr = np.sqrt(mean_squared_error(y_val, y_val_pred_lr))
print("Linear Regression RMSE:", rmse_lr)



# Initialize and train Random Forest
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_reg.fit(X_train, y_train)

# Predict on validation set
y_val_pred_rf = rf_reg.predict(X_val)

# Calculate RMSE
rmse_rf = np.sqrt(mean_squared_error(y_val, y_val_pred_rf))
print("Random Forest RMSE:", rmse_rf)




# Import necessary libraries
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Initialize XGBoost model with reasonable baseline parameters
xgb_reg = XGBRegressor(
    n_estimators=500,         # Number of trees (boosting rounds)
    learning_rate=0.05,       # Step size shrinkage used in each boosting step
    max_depth=6,              # Maximum tree depth for base learners
    subsample=0.8,            # Fraction of training samples used per tree
    colsample_bytree=0.8,     # Fraction of features used per tree
    random_state=42,          # Ensures reproducibility
    n_jobs=-1,                # Use all CPU cores
    objective='reg:squarederror'  # Regression objective for RMSE
)

#  Train model on the training set
xgb_reg.fit(X_train, y_train)

#  Predict on the validation set
y_val_pred_xgb = xgb_reg.predict(X_val)

#  Calculate RMSE for XGBoost
rmse_xgb = np.sqrt(mean_squared_error(y_val, y_val_pred_xgb))
print("XGBoost RMSE:", rmse_xgb)



from sklearn.model_selection import cross_val_score, KFold

# Define K-Fold strategy (5 folds here, you can adjust)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validation for Linear Regression
lin_reg = LinearRegression()
cv_scores_lr = cross_val_score(lin_reg, X_preprocessed, y, 
                               scoring="neg_root_mean_squared_error", cv=kf)
print("Linear Regression CV RMSE (per fold):", -cv_scores_lr)
print("Linear Regression Mean CV RMSE:", -cv_scores_lr.mean())

# Cross-validation for Random Forest
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
cv_scores_rf = cross_val_score(rf_reg, X_preprocessed, y, 
                               scoring="neg_root_mean_squared_error", cv=kf)
print("Random Forest CV RMSE (per fold):", -cv_scores_rf)
print("Random Forest Mean CV RMSE:", -cv_scores_rf.mean())





from sklearn.model_selection import cross_val_score, KFold

# Define K-Fold strategy (5 folds, same as before)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
cv_scores_xgb = cross_val_score(
    xgb_reg, X_preprocessed, y,
    scoring="neg_root_mean_squared_error", 
    cv=kf
)

# Since scores are negative (by sklearn convention), we invert them
print("XGBoost CV RMSE (per fold):", -cv_scores_xgb)
print("XGBoost Mean CV RMSE:", -cv_scores_xgb.mean())





import pandas as pd
import numpy as np

# Compute mean RMSEs for each model
rmse_lr_mean = -cv_scores_lr.mean()
rmse_rf_mean = -cv_scores_rf.mean()
rmse_xgb_mean = -cv_scores_xgb.mean()

# Create a DataFrame for comparison
results_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'XGBoost'],
    'Mean CV RMSE': [rmse_lr_mean, rmse_rf_mean, rmse_xgb_mean],
    'Std CV RMSE': [
        np.std(-cv_scores_lr),
        np.std(-cv_scores_rf),
        np.std(-cv_scores_xgb)
    ]
})

# Sort by performance (lower RMSE = better)
results_df = results_df.sort_values(by='Mean CV RMSE', ascending=True).reset_index(drop=True)

print(" Model Performance Comparison:")
print(results_df)



# # Train Random Forest on the full training set (so it's fitted properly)
# rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
# rf_reg.fit(X_preprocessed, y)

# # Predict accident_risk for test set
# test_preds = rf_reg.predict(X_test_preprocessed)

# # Clip predictions to stay within [0,1] (Kaggle requires this)
# test_preds = np.clip(test_preds, 0, 1)

# # Create submission dataframe
# submission_df = pd.DataFrame({
#     "id": test_df["id"],
#     "accident_risk": test_preds
# })

# # Save to CSV in correct format
# submission_df.to_csv("submission.csv", index=False)

# print("Submission file created: submission.csv")
# print(submission_df.head())



from xgboost import XGBRegressor
import numpy as np
import pandas as pd

#  Train XGBoost on the full training set (best-performing model)
xgb_reg = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

xgb_reg.fit(X_preprocessed, y)

#  Predict accident_risk for test set
test_preds = xgb_reg.predict(X_test_preprocessed)

#  Clip predictions to stay within [0,1] (important for Kaggle submission)
test_preds = np.clip(test_preds, 0, 1)

#  Create submission dataframe
submission_df = pd.DataFrame({
    "id": test_df["id"],
    "accident_risk": test_preds
})

#  Save to CSV in correct format
submission_df.to_csv("submission.csv", index=False)

print(" Submission file created: submission.csv")
print(submission_df.head())





