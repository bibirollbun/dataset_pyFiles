# =========================================================
# Predicting Road Accident Risk - Kaggle Playground S5E10
# =========================================================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# 2. Load Datasets
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
original = pd.read_csv("/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_100k.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Original shape:", original.shape)

# 3. Merge original + test for EDA
merged = pd.concat([original, test], ignore_index=True)
print("\nMerged shape:", merged.shape)

# 4. Missing Data Check & Imputation
num_cols = merged.select_dtypes(include=[np.number]).columns
cat_cols = merged.select_dtypes(include=['object', 'bool']).columns

# Numeric imputation
merged[num_cols] = merged[num_cols].fillna(merged[num_cols].mean())

# Categorical imputation
merged[cat_cols] = merged[cat_cols].apply(lambda x: x.fillna(x.mode()[0]))

print("\nMissing values after imputation:", merged.isnull().sum().sum())

# 5. Quick EDA Visualization
plt.figure(figsize=(8,5))
sns.countplot(data=merged, x='road_type', hue='public_road', palette='Set2')
plt.title('Road Type vs Public Road')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(data=original, x='weather', y='accident_risk', palette='coolwarm')
plt.title('Accident Risk by Weather')
plt.xticks(rotation=45)
plt.show()

# 6. Encode categorical variables for modeling
X = pd.get_dummies(original.drop(columns=['accident_risk']), drop_first=True)
y = original['accident_risk']

# Align training and test columns
test_encoded = pd.get_dummies(test, drop_first=True)
test_encoded = test_encoded.reindex(columns=X.columns, fill_value=0)

# 7. Split data for validation
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8. Define the XGBRegressor model
xgb = XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1
)

# 9. Define parameter grid for RandomizedSearchCV
param_dist = {
    'n_estimators': [300, 500, 700, 900],
    'max_depth': [3, 5, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.2, 0.3]
}

# 10. Hyperparameter Tuning
random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=25,
    scoring='neg_root_mean_squared_error',
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print("\nBest Parameters:", random_search.best_params_)
print("Best RMSE:", -random_search.best_score_)

# 11. Evaluate the Best Model on Validation Set
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_valid)

rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
r2 = r2_score(y_valid, y_pred)

print(f"\nValidation RMSE: {rmse:.4f}")
print(f"Validation R²: {r2:.4f}")

# 12. Retrain Best Model on Full Data
best_model.fit(X, y)

# 13. Predict on Test Set
preds = best_model.predict(test_encoded)

# 14. Prepare Submission
submission = sample_submission.copy()
submission['accident_risk'] = preds
submission['accident_risk'] = submission['accident_risk'].fillna(submission['accident_risk'].mean())




# 15. Save File
submission.to_csv("/kaggle/working/submission.csv", index=False)

submission.head(5)




