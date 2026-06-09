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


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import shap

# Load the data
train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")

# Encode the target
y = LabelEncoder().fit_transform(train["Personality"])

# Top correlated (leaky) features
leaky_features = [
    "Time_spent_Alone",
    "Drained_after_socializing",
    "Friends_circle_size",
    "Post_frequency",
    "Social_event_attendance",
    "Going_outside",
]

# Convert to numeric
train[leaky_features] = train[leaky_features].apply(pd.to_numeric, errors='coerce')
test[leaky_features] = test[leaky_features].apply(pd.to_numeric, errors='coerce')

# Frequency encode 'Post_frequency'
for col in ["Post_frequency"]:
    freq_map = train[col].value_counts(normalize=True).to_dict()
    train[col + "_freq"] = train[col].map(freq_map)
    test[col + "_freq"] = test[col].map(freq_map)

# Feature interactions
train["Interaction_1"] = train["Time_spent_Alone"] * train["Drained_after_socializing"]
test["Interaction_1"] = test["Time_spent_Alone"] * test["Drained_after_socializing"]
train["Interaction_2"] = train["Post_frequency"] * train["Friends_circle_size"]
test["Interaction_2"] = test["Post_frequency"] * test["Friends_circle_size"]
train["Interaction_3"] = train["Going_outside"] * train["Social_event_attendance"]
test["Interaction_3"] = test["Going_outside"] * test["Social_event_attendance"]

# --- Full Feature Set ---
X_full = pd.get_dummies(train.drop(columns=["id", "Personality"]))
X_test_full = pd.get_dummies(test.drop(columns=["id"]))
X_test_full = X_test_full.reindex(columns=X_full.columns, fill_value=0)

X_full = X_full.dropna(axis=1, how='all')
X_test_full = X_test_full[X_full.columns]

# --- Leak Feature Set ---
leak_features_plus = leaky_features + ["Post_frequency_freq", "Interaction_1", "Interaction_2", "Interaction_3"]
X_leak_raw = train[leak_features_plus].copy()
X_test_leak_raw = test[leak_features_plus].copy()
X_leak_raw = X_leak_raw.apply(pd.to_numeric, errors='coerce')
X_test_leak_raw = X_test_leak_raw.apply(pd.to_numeric, errors='coerce')
X_leak_raw = X_leak_raw.dropna(axis=1, how='all')
X_test_leak_raw = X_test_leak_raw[X_leak_raw.columns]

imputer_leak = SimpleImputer(strategy="mean")
X_leak_imputed = imputer_leak.fit_transform(X_leak_raw)
X_test_leak_imputed = imputer_leak.transform(X_test_leak_raw)
X_leak = pd.DataFrame(X_leak_imputed, columns=X_leak_raw.columns)
X_test_leak = pd.DataFrame(X_test_leak_imputed, columns=X_leak_raw.columns)

full_cols = X_full.columns.tolist()
imputer_full = SimpleImputer(strategy="mean")
X_full = pd.DataFrame(imputer_full.fit_transform(X_full), columns=full_cols)
X_test_full = pd.DataFrame(imputer_full.transform(X_test_full), columns=full_cols)

# --- Feature Selection via SHAP ---
model_shap = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.01, random_state=42)
model_shap.fit(X_full, y)
explainer = shap.TreeExplainer(model_shap)
shap_values = explainer.shap_values(X_full)
importance = np.abs(shap_values).mean(axis=1).mean(axis=0)
selected_features = np.array(full_cols)[importance > np.percentile(importance, 40)].tolist()
X_full = X_full[selected_features]
X_test_full = X_test_full[selected_features]

# --- Train/Validation Split ---
X_train, X_val, y_train, y_val = train_test_split(X_full, y, test_size=0.2, stratify=y, random_state=42)

# --- Model A: Full Model ---
model_full = lgb.LGBMClassifier(n_estimators=2500, learning_rate=0.005, max_depth=14, subsample=0.9,
                                colsample_bytree=0.85, class_weight="balanced", reg_alpha=1.0, reg_lambda=1.0,
                                feature_fraction_seed=42, random_state=42)
model_full.fit(X_train, y_train)
val_preds_full = model_full.predict(X_val)
val_probs_full = model_full.predict_proba(X_val)
print("ðŸ“Š Validation Accuracy - Full Model:", round(accuracy_score(y_val, val_preds_full), 5))

# --- Model B: Leak Model ---
model_leak = lgb.LGBMClassifier(n_estimators=3000, max_depth=12, learning_rate=0.02, subsample=0.9,
                                colsample_bytree=0.8, random_state=42)
model_leak.fit(X_leak, y)
val_probs_leak = model_leak.predict_proba(X_leak.iloc[X_val.index])

# --- Calibrate Full Model ---
calibrated_full = CalibratedClassifierCV(estimator=model_full, cv=5)
calibrated_full.fit(X_val, y_val)
val_probs_full_cal = calibrated_full.predict_proba(X_val)

# --- Meta Stacker ---
meta_input = np.hstack([val_probs_full_cal, val_probs_leak])
meta_model = LogisticRegression(max_iter=1000)
meta_model.fit(meta_input, y_val)

# --- Pseudo-labeling ---
probs_test = calibrated_full.predict_proba(X_test_full)
high_conf_mask = (np.max(probs_test, axis=1) - np.sort(probs_test, axis=1)[:, -2]) > 0.3
X_pseudo = X_test_full[high_conf_mask]
y_pseudo = probs_test[high_conf_mask].argmax(axis=1)

X_augmented = pd.concat([X_full, X_pseudo], axis=0)
y_augmented = np.concatenate([y, y_pseudo])

# --- Retrain Final Model ---
model_final = lgb.LGBMClassifier(n_estimators=2500, learning_rate=0.005, max_depth=14, subsample=0.9,
                                 colsample_bytree=0.85, class_weight="balanced", reg_alpha=1.0, reg_lambda=1.0,
                                 feature_fraction_seed=42, random_state=42)
model_final.fit(X_augmented, y_augmented)
final_probs_full = model_final.predict_proba(X_test_full)
final_probs_leak = model_leak.predict_proba(X_test_leak)

# --- Meta Prediction ---
test_meta_input = np.hstack([final_probs_full, final_probs_leak])
final_preds = meta_model.predict(test_meta_input)

# --- Submission ---
submission = sample_submission.copy()
submission["Personality"] = LabelEncoder().fit(train["Personality"]).inverse_transform(final_preds)
submission.to_csv("submission_blended_pseudolabel.csv", index=False)
print("âœ… Submission file saved as 'submission_blended_pseudolabel.csv'")





