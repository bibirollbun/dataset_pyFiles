# ==========================================
# ğŸš— Road Accident Risk Prediction
# ==========================================

# ----------------------------------
# 1ï¸�âƒ£ Importing neccessary packages
# ----------------------------------

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# ---------------------------
# 1ï¸�âƒ£ Load Data
# ---------------------------

train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
target_col = "accident_risk"
print("âœ… Data Loaded Successfully!")
print("Train shape:", train.shape)
print("Test shape:", test.shape)

plt.figure(figsize=(12,6))
sns.heatmap(train.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()

plt.figure(figsize=(7,5))
sns.histplot(train[target_col], bins=30, kde=True, color="teal")
plt.title("Target Distribution")
plt.xlabel(target_col)
plt.show()

plt.figure(figsize=(10,8))
numeric_cols = train.select_dtypes(include=['number']) 
sns.heatmap(numeric_cols.corr(), annot=False, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()


# -------------------------------
# 2ï¸�âƒ£ Prepare Features and Target
# -------------------------------

target_col = "accident_risk"
ID_col = "id"
X = train.drop(columns=[target_col, ID_col])
y = train[target_col]


# -------------------------------
# 3ï¸�âƒ£ Encode Categorical Columns
# -------------------------------

all_data = pd.concat([X, test.drop(columns=[ID_col])], axis=0)
categorical_cols = all_data.select_dtypes(include=["object", "bool"]).columns
encoder = LabelEncoder()
for col in categorical_cols:
    all_data[col] = encoder.fit_transform(all_data[col].astype(str))
X_encoded = all_data.iloc[:len(X), :]
test_encoded = all_data.iloc[len(X):, :]

encoded_all_data = all_data.copy()
for col in categorical_cols:
    encoded_all_data[col] = encoder.fit_transform(encoded_all_data[col].astype(str))
    
for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=all_data.iloc[:len(X)], x=col, order=all_data[col].value_counts().index)
    plt.title(f"Distribution of '{col}' (Before Encoding)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
for col in categorical_cols:
    plt.figure(figsize=(6, 6))
    encoded_all_data[col].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title(f"Pie Chart of Encoded '{col}'")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


# ---------------------------
# 3ï¸�âƒ£ Train/Validation Split
# ---------------------------

X_train, X_valid, y_train, y_valid = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

plt.figure(figsize=(6,4))
sns.kdeplot(y_train, label="Train", shade=True)
sns.kdeplot(y_valid, label="Validation", shade=True)
plt.title("Target Distribution: Train vs Validation")
plt.legend()
plt.show()



# ---------------------------
# 4ï¸�âƒ£ Train Ensemble Model
# ---------------------------

xgb_model = XGBRegressor(
    learning_rate=0.03,
    n_estimators=1000,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    tree_method="hist"
)

lgb_model = LGBMRegressor(
    learning_rate=0.03,
    n_estimators=1000,
    max_depth=-1,
    num_leaves=40,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)

cat_model = CatBoostRegressor(
    learning_rate=0.03,
    iterations=1000,
    depth=8,
    l2_leaf_reg=3,
    random_seed=42,
    verbose=0
)
stacked_model = StackingRegressor(
    estimators=[
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('cat', cat_model)
    ],
    final_estimator=RidgeCV(),
    n_jobs=-1,
    passthrough=False
)
print("ğŸš€ Training stacked ensemble ...")
stacked_model.fit(X_train, y_train)

print("âœ… Model training complete!")


# -----------------------------------
# 5ï¸�âƒ£ Evaluate Ensemble on Validation
# -----------------------------------

y_pred = stacked_model.predict(X_valid)
rmse = mean_squared_error(y_valid, y_pred, squared=False)
r2 = r2_score(y_valid, y_pred)
print(f"ğŸ“‰ Validation RMSE: {rmse:.6f}")
print(f"ğŸ“Š Validation RÂ² Score: {r2:.4f}")

# ---------------------------
# Plot 1: Actual vs Predicted
# ---------------------------
plt.figure(figsize=(6,6))
plt.scatter(y_valid, y_pred, alpha=0.5, color="tomato", edgecolor='k')
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted (Validation)")
plt.grid(True)
plt.show()

# ---------------------------
# Plot 2: Residuals Distribution
# ---------------------------
residuals = y_valid - y_pred
plt.figure(figsize=(7,4))
sns.histplot(residuals, bins=30, kde=True, color='purple')
plt.axvline(0, color='gray', linestyle='--')
plt.title("Residuals Distribution")
plt.xlabel("Residuals (Actual - Predicted)")
plt.ylabel("Frequency")
plt.show()


# ---------------------------
# 7ï¸�âƒ£ Predict on Test Data
# ---------------------------
test_preds = stacked_model.predict(test_encoded)
test_preds = np.clip(test_preds, 0, 1)


submission = pd.DataFrame({
    ID_col: test[ID_col],
    target_col: test_preds
})
submission.to_csv("submission.csv", index=False)
print(submission.head(10))

