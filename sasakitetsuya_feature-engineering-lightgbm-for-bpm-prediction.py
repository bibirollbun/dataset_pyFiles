
# === Setup & Imports ===
import os, glob
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

from lightgbm import LGBMRegressor




# === Data Loading ===
csv_path = "/kaggle/input/playground-series-s5e9/train.csv" 
df = pd.read_csv(csv_path)

print(df.shape)
print(df.dtypes)
assert "BeatsPerMinute" in df.columns, "Target column 'BeatsPerMinute' not found."




# Drop non-feature columns for correlation (keep the target just for display)
cols = [c for c in df.columns if c not in ["id"]]
corr = df[cols].corr()

# Show BPM correlations sorted
bpm_corr = corr["BeatsPerMinute"].sort_values(ascending=False)
print("Correlation w.r.t. BeatsPerMinute (descending):")
print(bpm_corr)

# Heatmap (matplotlib)
plt.figure(figsize=(8, 6))
cax = plt.imshow(corr.values, interpolation='nearest')
plt.title("Correlation Heatmap")
plt.xticks(range(len(cols)), cols, rotation=90)
plt.yticks(range(len(cols)), cols)
plt.colorbar(cax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()




X = df.drop(columns=["id", "BeatsPerMinute"], errors="ignore")
y = df["BeatsPerMinute"].astype(float)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LGBMRegressor(
    n_estimators=400,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="rmse")

preds = model.predict(X_val)
rmse = mean_squared_error(y_val, preds, squared=False)
print(f"Baseline RMSE: {rmse:.4f}")

# Feature importance
importances = model.feature_importances_
features = np.array(X.columns)

order = np.argsort(importances)[::-1]
plt.figure(figsize=(7, 5))
plt.barh(range(len(features)), importances[order])
plt.yticks(range(len(features)), features[order])
plt.gca().invert_yaxis()
plt.title("LightGBM Feature Importance (Baseline)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()




poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_inter = poly.fit_transform(X)
feature_names = poly.get_feature_names_out(X.columns)
X_poly = pd.DataFrame(X_inter, columns=feature_names)

X_train2, X_val2, y_train2, y_val2 = train_test_split(
    X_poly, y, test_size=0.2, random_state=42
)

model2 = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1
)
model2.fit(X_train2, y_train2, eval_set=[(X_val2, y_val2)], eval_metric="rmse")

preds2 = model2.predict(X_val2)
rmse2 = mean_squared_error(y_val2, preds2, squared=False)
print(f"With Interactions RMSE: {rmse2:.4f}")

imp2 = model2.feature_importances_
order2 = np.argsort(imp2)[::-1]
topk = 20 if len(order2) >= 20 else len(order2)

plt.figure(figsize=(9, 7))
plt.barh(range(topk), imp2[order2][:topk])
plt.yticks(range(topk), np.array(feature_names)[order2][:topk])
plt.gca().invert_yaxis()
plt.title("Top-20 Feature Importances (LightGBM with Pairwise Interactions)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()


