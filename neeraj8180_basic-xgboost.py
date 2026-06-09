# =======================
# 0. IMPORTS
# =======================
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# =======================
# 1. LOAD DATA
# =======================
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')

target_col = 'Personality'
id_col = 'id'

X_train = train.drop([target_col], axis=1)
y_train_raw = train[target_col]
X_test = test.copy()

# =======================
# 2. ENCODE TARGET
# =======================
label_encoder_target = LabelEncoder()
y_train = label_encoder_target.fit_transform(y_train_raw)

print('Target classes:', label_encoder_target.classes_)

# =======================
# 3. DETECT COLUMNS
# =======================
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

if id_col in num_cols:
    num_cols.remove(id_col)

# =======================
# 4. SIMPLE IMPUTATION
# =======================
filled_train = X_train.copy()
filled_test = X_test.copy()

for col in num_cols:
    mean_val = filled_train[col].mean()
    filled_train[col] = filled_train[col].fillna(mean_val)
    filled_test[col] = filled_test[col].fillna(mean_val)

for col in cat_cols:
    mode_val = filled_train[col].mode()[0]
    filled_train[col] = filled_train[col].fillna(mode_val)
    filled_test[col] = filled_test[col].fillna(mode_val)

# =======================
# 5. LABEL ENCODE CATEGORICALS
# =======================
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([filled_train[col], filled_test[col]], axis=0).astype(str)
    le.fit(combined)
    filled_train[col] = le.transform(filled_train[col].astype(str))
    filled_test[col] = le.transform(filled_test[col].astype(str))

# =======================
# 6. FEATURE ENGINEERING (USING ACTUAL COLUMNS)
# =======================
# Create new features based on your actual columns
filled_train['Social_Activity_Score'] = filled_train['Social_event_attendance'] + filled_train['Going_outside']
filled_test['Social_Activity_Score'] = filled_test['Social_event_attendance'] + filled_test['Going_outside']

filled_train['Energy_Drain_Ratio'] = filled_train['Drained_after_socializing'] / (filled_train['Friends_circle_size'] + 1)
filled_test['Energy_Drain_Ratio'] = filled_test['Drained_after_socializing'] / (filled_test['Friends_circle_size'] + 1)

filled_train['Social_Engagement'] = filled_train['Post_frequency'] * filled_train['Friends_circle_size']
filled_test['Social_Engagement'] = filled_test['Post_frequency'] * filled_test['Friends_circle_size']

# =======================
# 7. FINAL FEATURES
# =======================
features = [c for c in filled_train.columns if c != id_col]
X_train_final = filled_train[features]
X_test_final = filled_test[features]

print(f"\nNumber of features used: {len(features)}")
print("Features:", features)

# =======================
# 8. TRAIN-VALID SPLIT
# =======================
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_final, y_train, test_size=0.2, stratify=y_train, random_state=42
)

# =======================
# 9. XGBOOST
# =======================
xgb_clf = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
xgb_clf.fit(X_tr, y_tr)

# =======================
# 10. VALIDATION METRICS
# =======================
y_val_pred = xgb_clf.predict(X_val)
val_acc = accuracy_score(y_val, y_val_pred)
print(f"\nValidation Accuracy: {val_acc:.4f}\n")
print("Classification Report:\n", classification_report(y_val, y_val_pred, target_names=label_encoder_target.classes_))

# Plot feature importance
plt.figure(figsize=(10, 6))
xgb.plot_importance(xgb_clf, max_num_features=15)
plt.title('Feature Importance')
plt.show()

# =======================
# 11. PREDICT TEST SET
# =======================
test_preds_numeric = xgb_clf.predict(X_test_final)
test_preds = label_encoder_target.inverse_transform(test_preds_numeric)

# =======================
# 12. SUBMISSION
# =======================
submission = pd.DataFrame({
    id_col: test[id_col],
    target_col: test_preds
})
submission.to_csv('submission.csv', index=False)
print("\nsubmission.csv generated!")




