import pandas as pd
import numpy as np

train = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/train.csv')
test = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/test.csv')
sample = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/sample_submission.csv')

print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


import seaborn as sns
import matplotlib.pyplot as plt

# Target Distribution
sns.histplot(train['Lap_Time_Seconds'], kde=True, bins=30)
plt.title("Target Distribution - Lap Time (Seconds)")
plt.show()

# Null Count
print("Missing values:\n", train.isnull().sum().sort_values(ascending=False).head(10))


numeric_cols = train.select_dtypes(include=['int64', 'float64']).columns
corr = train[numeric_cols].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr[['Lap_Time_Seconds']].sort_values(by='Lap_Time_Seconds', ascending=False), annot=True, cmap='coolwarm')
plt.title("Correlation with Lap_Time_Seconds")
plt.show()


!pip install --quiet xgboost==2.0.3 pandas==2.2.2

import pandas as pd, numpy as np, os, xgboost as xgb, time, gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Paths & column names
DATA_DIR = "/kaggle/input/burnout-datathon-ieeecsmuj"   
TARGET   = "Lap_Time_Seconds"                
ID_COL   = "Unique ID"                       

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
print("Train shape:", train_df.shape, "  Test shape:", test_df.shape)


# Minimal feature engineering
def add_features(df):
    if {"Circuit_Length_km", "Avg_Speed_kmh"}.issubset(df.columns):
        df["secs_per_km"] = (df["Circuit_Length_km"] / df["Avg_Speed_kmh"]) * 3600
    if {"Track_Temperature_Celsius", "Ambient_Temperature_Celsius"}.issubset(df.columns):
        df["track_minus_air"] = df["Track_Temperature_Celsius"] - df["Ambient_Temperature_Celsius"]
    if "Track_Condition" in df.columns:
        df["is_wet"] = df["Track_Condition"].str.contains("Wet", case=False, na=False).astype(int)
    if {"Tire_Compound_Front", "Tire_Compound_Rear"}.issubset(df.columns):
        df["tire_combo"] = df["Tire_Compound_Front"].fillna("") + "_" + df["Tire_Compound_Rear"].fillna("")
    return df

train_df = add_features(train_df);  test_df = add_features(test_df)


# Convert position numeric → categorical string (if present)
if "position" in train_df.columns:
    train_df["Position_cat"] = train_df["position"].astype(str)
    test_df["Position_cat"]  = test_df["position"].astype(str)
    train_df.drop(columns=["position"], inplace=True)
    test_df.drop(columns=["position"],  inplace=True)


# Identify categorical & numeric columns robustly
cat_cols = [c for c in train_df.columns
            if train_df[c].dtype == "object" or train_df[c].dtype.name == "category"]
num_cols = [c for c in train_df.columns if c not in cat_cols + [TARGET]]

# Keep only cats present in both frames
cat_cols = [c for c in cat_cols if c in test_df.columns]

# Fill NaNs
train_df[cat_cols] = train_df[cat_cols].fillna("missing")
test_df[cat_cols]  = test_df[cat_cols].fillna("missing")
for col in num_cols:
    med = train_df[col].median()
    train_df[col].fillna(med, inplace=True)
    test_df[col].fillna(med,  inplace=True)

# Cast categorical dtype for XGB native cats
for c in cat_cols:
    train_df[c] = train_df[c].astype("category")
    test_df[c]  = test_df[c].astype("category")

feature_cols = cat_cols + num_cols
print(f"Using {len(feature_cols)} features ({len(cat_cols)} categorical).")


# Train‑validation split 
X_train, X_val, y_train, y_val = train_test_split(
    train_df[feature_cols], train_df[TARGET],
    test_size=0.2, random_state=42
)

dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
dval   = xgb.DMatrix(X_val,   label=y_val,   enable_categorical=True)
dtest  = xgb.DMatrix(test_df[feature_cols], enable_categorical=True)


# XGBoost parameters & training 
params = {
    "objective": "reg:squarederror", # Objective for regression tasks (predicting continuous values)
    "eval_metric": "rmse",           # Evaluation metric: Root Mean Squared Error
    "learning_rate": 0.05,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "lambda": 3.0,
    "tree_method": "gpu_hist",       # Use GPU for faster training if available
    "enable_categorical": True,      # Enable native categorical feature handling
    "seed": 42,
}

start = time.time()
model = xgb.train(
    params,
    dtrain, # Training data (DMatrix)
    num_boost_round=5000, # Max boosting rounds
    evals=[(dval, "val")], # Validation set for early stopping
    early_stopping_rounds=200, # Stop if validation RMSE doesn't improve for 200 rounds
    verbose_eval=250 # Print progress every 250 rounds
)

print(f"Training time: {time.time()-start:.1f} s")
print("Best iteration:", model.best_iteration)
print("Best val RMSE:", model.best_score)


# Predict on test & create submission
test_preds = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
submission = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET: test_preds})
submission.to_csv("submission_xgb.csv", index=False)
print("submission_xgb.csv saved:", submission.shape)
submission.head()




