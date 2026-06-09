# Install common packages (Kaggle usually has them, but safe to include)
!pip install -q pandas numpy scikit-learn pyarrow


import os
import glob
import zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# List top-level input folders mounted by Kaggle
print("Listing /kaggle/input contents:")
!ls -la /kaggle/input


# Find any train.csv under /kaggle/input (search recursively)
csv_paths = glob.glob("/kaggle/input/**/train.csv", recursive=True)

if len(csv_paths) == 0:
    raise FileNotFoundError("No train.csv found under /kaggle/input. Make sure you added the dataset to the notebook.")
else:
    # If multiple train.csv files exist, pick the first (you can change index if needed)
    train_path = csv_paths[0]
    print("Using train file:", train_path)

# Load dataframe
d = pd.read_csv(train_path)
print("\nLoaded dataframe shape:", d.shape)
display(d.head())
print("\nColumns:", list(d.columns))



# If there's no explicit 'id' column, create one from the original row index
if 'id' not in d.columns:
    d = d.reset_index().rename(columns={'index': 'id'})
    print("No 'id' column found — created 'id' from index.")
else:
    print("'id' column found and will be preserved.")

# Quick info
print("\nData types and non-null counts:")
display(d.info())
print("\nMissing values per column:")
display(d.isnull().sum())


# Fill numeric NAs with column mean (only numeric columns)
numeric_cols = d.select_dtypes(include=[np.number]).columns.tolist()
d[numeric_cols] = d[numeric_cols].fillna(d[numeric_cols].mean())

# If 'num_lanes' exists, categorize it; if not, skip but warn
if 'num_lanes' in d.columns:
    def lane_category(x):
        # adjust bins as per your domain knowledge
        if x <= 2:
            return "Low"
        elif x <= 4:
            return "Medium"
        elif x <= 6:
            return "High"
        else:
            return "Very_High"
    d['lane_category'] = d['num_lanes'].apply(lane_category)
    # keep original num_lanes if you want; if you prefer remove it uncomment next line:
    # d.drop(columns=['num_lanes'], inplace=True)
    print("Categorized 'num_lanes' into 'lane_category'.")
else:
    print("Warning: 'num_lanes' column not found — skipping lane categorization.")

# If any remaining nulls (non-numeric) — drop rows (or you can impute differently)
if d.isnull().any().any():
    print("Warning: Some non-numeric columns still have missing values. Dropping rows with NA.")
    d = d.dropna().reset_index(drop=True)

print("\nAfter cleaning shape:", d.shape)
display(d.head())



# Identify categorical columns (object or bool or category) excluding target and id
exclude = {'id', 'accident_risk'}  # keep id & target separate
cat_cols = [c for c in d.select_dtypes(include=['object', 'bool', 'category']).columns if c not in exclude]
print("Categorical columns to encode:", cat_cols)

# One-hot encode (drop_first=True to avoid dummy trap)
d_encoded = pd.get_dummies(d, columns=cat_cols, drop_first=True)

print("\nEncoded dataframe shape:", d_encoded.shape)
display(d_encoded.head())



# Ensure target exists
if 'accident_risk' not in d_encoded.columns:
    raise KeyError("Target column 'accident_risk' not found in data. Ensure train.csv contains this column.")

X = d_encoded.drop(columns=['accident_risk', 'id'], errors='ignore')
y = d_encoded['accident_risk']

# 80% train, 20% validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)



from sklearn.ensemble import RandomForestRegressor

# Simplified & faster Random Forest model
rf = RandomForestRegressor(
    n_estimators=200,      
    max_features='sqrt',   # good balance for performance
    max_depth=15,          # limit tree depth to avoid overfitting and speed up
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1              # use all cores
)

print("⏳ Training Random Forest model...")
rf.fit(X_train, y_train)
print("✅ Random Forest training completed successfully!")


y_val_pred = rf.predict(X_val)
r2 = r2_score(y_val, y_val_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f"Validation R²: {r2:.4f}")
print(f"Validation RMSE: {rmse:.6f}")

# Quick scatter plot: actual vs predicted
plt.figure(figsize=(6,5))
plt.scatter(y_val, y_val_pred, alpha=0.4)
plt.xlabel("Actual accident_risk")
plt.ylabel("Predicted accident_risk")
plt.title("Actual vs Predicted (validation)")
plt.plot([0,1],[0,1], color='red', linestyle='--')
plt.show()



# Show top 15 features
importances = pd.Series(rf.feature_importances_, index=X_train.columns)
top_features = importances.sort_values(ascending=False).head(15)
plt.figure(figsize=(8,6))
top_features.plot(kind='barh')
plt.gca().invert_yaxis()
plt.title("Top 15 Feature Importances")
plt.xlabel("Importance")
plt.show()



# Prepare X_all that matches training columns (use d_encoded rows order)
X_all = d_encoded.drop(columns=['accident_risk', 'id'], errors='ignore')

# Ensure columns align with model training features (fill missing with 0 if any)
X_all = X_all.reindex(columns=X_train.columns, fill_value=0)

# Predict for all rows
all_preds = rf.predict(X_all)
all_preds = np.clip(all_preds, 0.0, 1.0)  # keep probabilities (or scores) between 0 and 1

# Build submission dataframe using original 'id' column
submission = pd.DataFrame({
    'id': d['id'],
    'accident_risk': all_preds
})

# Confirm row count equals original
print("Original rows:", d.shape[0], "Submission rows:", submission.shape[0])

# Save CSV and Parquet
submission.to_csv("submission.csv", index=False)
submission.to_parquet("submission.parquet", index=False)
print("Saved submission.csv and submission.parquet")
display(submission.head())



zip_name = "submission.zip"
with zipfile.ZipFile(zip_name, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write("submission.csv")
print(f"Created {zip_name} containing submission.csv")


