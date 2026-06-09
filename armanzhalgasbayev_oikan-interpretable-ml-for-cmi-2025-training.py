!pip install -qU oikan


!pip freeze | grep oikan


# === Imports ===
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif

from metric import score, CompetitionMetric  # Given competition metric
from oikan import OIKANClassifier


import warnings
warnings.filterwarnings('ignore')


# === CONFIG ===
TRAINING = 1
SAVING = 1


# === Data Loading ===
train = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
train_demographics = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")

train_merged = pd.merge(train, train_demographics, on="subject", how="left")


train_merged.head()


train_merged.info()


%%time

# === Feature Engineering ===
acc_cols = [f'acc_{axis}' for axis in ['x', 'y', 'z']]
rot_cols = [f'rot_{axis}' for axis in ['w', 'x', 'y', 'z']]
thm_cols = [f'thm_{i}' for i in range(1, 6)]
tof_cols = [f'tof_{i}_v{j}' for i in range(1, 6) for j in range(64)]

numerical_cols = ["age", "height_cm", "shoulder_to_wrist_cm", "elbow_to_wrist_cm"] + acc_cols + rot_cols + thm_cols + tof_cols
categorical_cols = ["adult_child", "sex", "handedness"]

summary_cols = ['sequence_id'] + acc_cols + rot_cols + thm_cols + tof_cols
summary = train_merged[summary_cols].groupby('sequence_id').agg(['mean', 'std', 'min', 'max', 'median'])

summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
summary = summary.reset_index()

remaining = train_merged.drop_duplicates('sequence_id')
train_df = pd.merge(remaining, summary, on="sequence_id", how="left")

exclude_cols = set(categorical_cols) | {"sequence_id"}
numerical_cols = [
    col for col in train_df.select_dtypes(include=["number"]).columns
    if col not in exclude_cols and col != "gesture"
]
X_full = train_df[numerical_cols]


# Label encode target
le = LabelEncoder()
y = le.fit_transform(train_df["gesture"])


# Mapping info:
labels_dict = {index: value for index, value in enumerate(le.classes_)}
print(labels_dict)


# Visualize Class Distribution
gesture_counts = train_df['gesture'].value_counts(normalize=True)
plt.figure(figsize=(8, 8))
plt.pie(gesture_counts, labels=gesture_counts.index, autopct='%1.1f%%', startangle=90)
plt.title("Gesture Class Distribution")
plt.axis('equal')
plt.show()


# Fill missing values with median (numeric only)
X = X_full.fillna(X_full.median(numeric_only=True))

# Standardize (Z-score normalization)
means = X.mean()
stds = X.std(ddof=0)
X_scaled = (X - means) / stds

X_scaled = X_scaled.fillna(0)


selector = SelectKBest(f_classif, k=300)  # try different k values
X_selected = selector.fit_transform(X_scaled, y)
selected_features = X.columns[selector.get_support()]
print(selected_features)


X = X_scaled[selected_features]


# === OIKAN Training ===
oikan_model = OIKANClassifier(
    augmentation_factor=1,
    alpha=0.2,
    top_k=30,
    verbose=True,
    random_state=42
)


%%time

if TRAINING:
    oikan_model.fit(X, y)


# === Display symbolic formula and importances ===
print("\n=== OIKAN Symbolic Formula ===")
formulas = oikan_model.get_formula()
for formula in formulas:
    print(formula)

print("\n=== OIKAN Feature Importances ===")
importances = oikan_model.feature_importances()
feature_names = selected_features

# Top 20 plot
plt.figure(figsize=(10, 6))
sorted_idx = np.argsort(importances)[::-1][:20]
plt.barh([feature_names[i] for i in sorted_idx][::-1], importances[sorted_idx][::-1])
plt.xlabel("Importance")
plt.title("Top 20 OIKAN Feature Importances")
plt.tight_layout()
plt.show()


# === Save Artifacts ===
if SAVING:
    os.makedirs("model_artifacts", exist_ok=True)

    # Save selected feature names for inference
    with open("model_artifacts/used_features.txt", "w") as f:
        f.write("\n".join(selected_features))

    # Save label encoder
    joblib.dump(le, "model_artifacts/label_encoder.pkl")

    # Save trained OIKAN model
    oikan_model.save("model_artifacts/oikan_model.json")

