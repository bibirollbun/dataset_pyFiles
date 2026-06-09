import numpy as np
import pandas as pd
df = pd.read_csv("/kaggle/input/diabetespredict/train.csv")

df


# ===================== ONE-CELL COMPLETE PIPELINE =====================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# ------------------ Load Data ------------------


# ------------------ Feature Selection ------------------
features = [
    'age',
    'alcohol_consumption_per_week',
    'physical_activity_minutes_per_week',
    'diet_score',
    'sleep_hours_per_day',
    'cholesterol_total',
    'bmi',
    'waist_to_hip_ratio',
    'gender',
    'smoking_status',
    'family_history_diabetes',
    'hypertension_history'
]

target = 'diagnosed_diabetes'

X = df[features].copy()
y = df[target]

# ------------------ Encode Categorical ------------------
label_encoders={}
for col in ['gender', 'smoking_status']:
    # X[col] = LabelEncoder().fit_transform(X[col])
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# ------------------ Train-Test Split ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------ Scaling ------------------
num_cols = [
    'age',
    'alcohol_consumption_per_week',
    'physical_activity_minutes_per_week',
    'diet_score',
    'sleep_hours_per_day',
    'cholesterol_total',
    'bmi',
    'waist_to_hip_ratio'
]

scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# ------------------ Models ------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=-1),
    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_split=20,
        random_state=42,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}

# ------------------ Train & Evaluate ------------------
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_pred_proba)
    print(f"{name} ROC-AUC: {roc:.4f}")

# ======================================================



df_test=pd.read_csv('/kaggle/input/diabetespredict/test.csv')


df_test


# =========================
# TEST.CSV PREDICTION CODE


# Select same features
X_test_final = df_test[features].copy()

# Encode categorical columns using trained encoders
for col in ['gender', 'smoking_status']:
    X_test_final[col] = label_encoders[col].transform(X_test_final[col])

# Scale numerical columns using trained scaler
X_test_final[num_cols] = scaler.transform(X_test_final[num_cols])

# Predict probabilities
test_probs = model.predict_proba(X_test_final)[:, 1]

# Create submission file
submission = pd.DataFrame({
    "id": df_test["id"],
    "diagnosed_diabetes": test_probs
})

submission.to_csv("submission.csv", index=False)
submission




