# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Smart Crop Recommendation - end-to-end example
# Run in Jupyter / Colab. Requires: pandas, numpy, scikit-learn, matplotlib, joblib, seaborn (optional)
# pip install pandas numpy scikit-learn matplotlib joblib seaborn

import os
import pandas as pd
import numpy as np

# ML imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# plotting (optional)
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# 1) Load dataset (or create small example if not found)
# ---------------------------
DATA_PATH = "crop_data.csv"  # replace with your dataset path

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    print("Loaded dataset:", DATA_PATH)
else:
    # Create a small synthetic example with typical columns used in many crop datasets
    print("Dataset not found. Creating a small synthetic example 'df' to demonstrate pipeline.")
    data = {
        "N":    [90, 85, 40, 20, 12, 50, 70, 10, 25, 80],
        "P":    [42, 58, 40, 42, 14, 40, 45, 10, 30, 60],
        "K":    [43, 35, 40, 40, 10, 30, 38, 5, 20, 42],
        "temperature": [28, 26, 27, 18, 15, 25, 29, 20, 21, 27],
        "humidity":    [80, 70, 65, 50, 40, 68, 75, 60, 55, 72],
        "ph":          [6.5,6.8,6.0,5.5,5.6,6.2,6.4,5.2,5.8,6.6],
        "rainfall":    [200, 150, 100, 60, 20, 120, 180, 40, 80, 160],
        "label": [
            "rice","rice","maize","wheat","sorghum",
            "maize","rice","sorghum","wheat","rice"
        ]
    }
    df = pd.DataFrame(data)
    # save a copy so user can inspect
    df.to_csv("crop_data_sample.csv", index=False)
    print("Saved sample data to 'crop_data_sample.csv'")

df.head()

# ---------------------------
# 2) Quick EDA (summary)
# ---------------------------
print("\nDataset shape:", df.shape)
print("\nColumns and types:\n", df.dtypes)
print("\nLabel value counts:\n", df['label'].value_counts())

# Optional: pairplot / correlations
plt.figure(figsize=(8,6))
sns.heatmap(df.drop(columns=['label']).corr(), annot=True, fmt=".2f")
plt.title("Feature Correlation")
plt.show()

# ---------------------------
# 3) Preprocessing
# ---------------------------
# Features & target
FEATURES = ["N","P","K","temperature","humidity","ph","rainfall"]
TARGET = "label"

X = df[FEATURES].values
y = df[TARGET].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (tree models don't require it but scaling helps some models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, "scaler.joblib")

# ---------------------------
# 4) Model training - Random Forest (with simple tuning)
# ---------------------------
rf = RandomForestClassifier(random_state=42, n_jobs=-1)

# Simple grid search (small search to keep quick)
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

grid = GridSearchCV(rf, param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=1)
grid.fit(X_train_scaled, y_train)

print("Best params:", grid.best_params_)
best_model = grid.best_estimator_

# ---------------------------
# 5) Evaluation
# ---------------------------
y_pred = best_model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
plt.show()

# ---------------------------
# 6) Save model
# ---------------------------
joblib.dump(best_model, "smart_crop_rf.joblib")
print("Saved model to 'smart_crop_rf.joblib' and scaler to 'scaler.joblib'")

# ---------------------------
# 7) Helper: prediction function for new samples
# ---------------------------
def recommend_crop(sample_dict):
    """
    sample_dict should contain keys: N, P, K, temperature, humidity, ph, rainfall
    Example:
      sample = {"N":90,"P":42,"K":43,"temperature":28,"humidity":80,"ph":6.5,"rainfall":200}
    """
    required = FEATURES
    # build array in expected order
    arr = np.array([[ sample_dict[k] for k in required ]], dtype=float)
    # load scaler & model (if not in memory)
    scaler_local = joblib.load("scaler.joblib")
    model_local = joblib.load("smart_crop_rf.joblib")
    arr_scaled = scaler_local.transform(arr)
    pred = model_local.predict(arr_scaled)
    proba = None
    if hasattr(model_local, "predict_proba"):
        proba = model_local.predict_proba(arr_scaled)
    return {"recommended_crop": pred[0], "probabilities": proba[0].tolist() if proba is not None else None}

# Example usage
example = {"N":90,"P":42,"K":43,"temperature":28,"humidity":80,"ph":6.5,"rainfall":200}
print("\nExample input:", example)
print("Recommendation:", recommend_crop(example))

# ---------------------------
# 8) Next steps / tips
# ---------------------------
print("""
TIPS:
- Replace 'crop_data.csv' with your real Kaggle dataset. Ensure column names match FEATURES.
- If you have class imbalance, consider stratified sampling or class_weight adjustments.
- For production: wrap the recommend_crop function in a Flask/FastAPI endpoint or use Gradio for a quick demo.
- You can augment input features (satellite NDVI, soil EC, historical yields, market price) for better recommendations.
""")


