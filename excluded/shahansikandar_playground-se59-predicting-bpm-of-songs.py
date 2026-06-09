# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb


# Load data
train_df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")


# Print first 5 rows
train_df.head()


# Drop 'id' column
train_df = train_df.drop(columns=["id"])


# Print shape
train_df.shape


# Display info
train_df.info()


# Check for missing values
train_df.isnull().sum()


# Descriptive statistics
train_df.describe().T


# Target 'BeatsPerMinute' distribution
plt.figure(figsize=(10,5))
sns.histplot(train_df['BeatsPerMinute'], bins=50, kde=True, color="purple")
plt.axvline(train_df['BeatsPerMinute'].mean(), color="red", linestyle="--", label="Mean BPM")
plt.legend()
plt.title("Distribution of BPM")
plt.show()



# Features distribution
features = ['RhythmScore','AudioLoudness','VocalContent',
            'AcousticQuality','InstrumentalScore','LivePerformanceLikelihood',
            'MoodScore','TrackDurationMs','Energy']

for col in features:
    plt.figure(figsize=(8,4))
    sns.histplot(train_df[col], bins=40, kde=True, color="teal")
    plt.title(f"Distribution of {col}")
    plt.show()



# Correlation matrix
plt.figure(figsize=(8, 5))
corr = train_df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Feature Correlations with BPM")
plt.show()


# Correlation values with BeatsPerMinute
corr_with_target = corr["BeatsPerMinute"].sort_values(ascending=False)
print("Correlation of features with BPM:\n", corr_with_target)


# Detect Outliers in features using BoxPlot
plt.figure(figsize=(15, 4 * len(features)))  # adjust height dynamically
for i, col in enumerate(features, 1):
    plt.subplot(len(features), 1, i)
    sns.boxplot(x=train_df[col], color="skyblue")
    plt.title(f"Boxplot of {col}", fontsize=12)
plt.tight_layout()
plt.show()



# Remove outlier with Z-score
z_scores = np.abs(stats.zscore(train_df[features]))
train_df_clean = train_df[(z_scores < 3).all(axis=1)]

print(f"Original shape: {train_df.shape}")
print(f"After removing outliers: {train_df_clean.shape}")



# Features
X = train_df_clean.drop(columns=["BeatsPerMinute", "id"], errors='ignore')

# Target
y = train_df_clean["BeatsPerMinute"]

# Split into train/validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



# Train XGBoost Regressor
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    n_jobs=-1,
    tree_method='hist',  
    random_state=42
)

# Fit the model
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_val)
print("XGBoost RMSE:", np.sqrt(mean_squared_error(y_val, y_pred_xgb)))



# Train LightGBM Regressor 
lgb_model = lgb.LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=8,
    random_state=42
)

# Fit the model
lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_val)
print("LightGBM RMSE:", np.sqrt(mean_squared_error(y_val, y_pred_lgb)))



# Feature importance
feat_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importance': lgb_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12,6))
sns.barplot(x='Importance', y='Feature', data=feat_imp)
plt.title("Feature Importance - LightGBM")
plt.show()



# Predict on Test Set & Submission

test_features = test_df.drop(columns=["id"], errors='ignore')

# Choose best model (LightGBM)
test_predictions = lgb_model.predict(test_features)

# Create submission DataFrame
submission = pd.DataFrame({
    "id": test_df["id"],
    "BeatsPerMinute": test_predictions
})

# Print first 10 rows of submission
print(submission.head(10))

# Save submission for judging
submission.to_csv("submission.csv", index=False)
print("Submission file saved!")


